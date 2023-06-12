import yaml
import numpy as np
import os
import soundfile as sf
from scipy.signal import stft
import torch
# If you have a GPU, put the data on the GPU
device = torch.device('cpu')

directory = "Data/slakh2100_flac_redux/train"
listdir = os.listdir(directory)
listdir.sort()
format = '.flac'
sample_freq = 44100
freq_amount = 129
mixing = True
mix_amount = 2
np.random.seed(0)
amount = 100
L = freq_amount
C = 2
NUMBER_OF_ITERATIONS = 2


def data_dicts(N: int = 10, directory=directory, format=format, sample_freq=sample_freq, freq_amount=freq_amount, mixing=mixing, mix_amount=mix_amount, device=device, dict1=False, all: bool = False):
    if all:
        iterator = np.random.shuffle(os.listdir(directory))
    else:
        iterator = np.random.choice(os.listdir(directory), N)
    for tr in iterator:
        tr_dict = {}
        tr_path = os.path.join(directory, tr)
        with open(os.path.join(tr_path, "metadata.yaml")) as meta:
            metadata = yaml.safe_load(meta)

        file_inst = np.array([[stem + format, metadata['stems'][stem]['inst_class']] for stem in metadata['stems'].keys()])
        file_inst = file_inst[[file_inst[:,0][i] in os.listdir(os.path.join(tr_path, 'stems')) for i in range(len(file_inst))]]
        # combine all stems from the same instrument and track using soundfile
        for inst in np.unique(file_inst[:,1]):
            inst_files = file_inst[file_inst[:,1] == inst][:,0]
            inst_files = [os.path.join(tr_path, 'stems', inst_file) for inst_file in inst_files]
            inst_data = np.array([sf.read(inst_file)[0] for inst_file in inst_files])
            inst_data = np.sum(inst_data, axis=0)
            # add the combined data to the dictionary
            tr_dict.update({inst: inst_data})
        
        
        # Choose a random amount of instruments to combine (between 2 and lenght-1)
        if mixing:
            # Find all instruments that are in the dictionary
            inst_in_dict = list(tr_dict.keys())
            inst_amount = np.random.randint(2, len(inst_in_dict), size=mix_amount)
            for i in range(mix_amount):
                # Choose the instruments to combine
                inst_to_mix = np.random.choice(inst_in_dict, inst_amount[i], replace=False)
                # Add the combined data to the dictionary
                tr_dict.update({''.join(inst_to_mix):  np.sum([tr_dict[inst] for inst in inst_to_mix], axis=0)})

        # Add mix to the dictionary
        inst_data = sf.read(os.path.join(tr_path, 'mix' + format))[0]
        tr_dict.update({'mix': inst_data})

        if dict1:
            yield tr_dict
        else:
            tr_dicts_2 = {inst: torch.view_as_real_copy(torch.from_numpy(stft(tr_dict[inst], fs=sample_freq, nperseg=freq_amount*2-2)[2])).to(device) for inst in tr_dict.keys()}
            yield tr_dicts_2
            del tr_dicts_2
        del tr_dict
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        
        
        
def data_frame(NUMBER_OF_ITERATIONS,amount, C = C, L = L, device=device, all: bool = False, **kwargs):
    for data in data_dicts(NUMBER_OF_ITERATIONS, freq_amount=L, device= device, all=all, **kwargs):
        dat = {}
        lenght = data['mix'].shape[1]
        for inst in data.keys():
            dat[inst] = torch.zeros(amount,L,2*C+1,2, dtype=torch.float64).to(device)
            for j, rand in enumerate(torch.randint(0, lenght, (amount,))):
                if rand < C:
                    dat[inst][j][:,0] = torch.zeros(L, 2).to(device)
                    i = 1
                    while rand-C+i < 0:
                        dat[inst][j][:,i] = torch.zeros(L, 2).to(device)
                        i+=1
                    dat[inst][j][:,i:] = data[inst][:, 0:rand+C+1]
                elif rand > lenght-1-C:
                    dat[inst][j][:,-1] = torch.zeros(L, 2).to(device)
                    i = 1
                    while rand+C-i > lenght-1:
                        dat[inst][j][:,-1-i] = torch.zeros(L, 2).to(device)
                        i+=1
                    dat[inst][j][:,:-1-i] = data[inst][:, rand-C:-1]
                else:
                    dat[inst][j] = data[inst][:, rand-C:rand+C+1]
        yield dat
        del dat
        if device == 'cuda':
            torch.cuda.empty_cache()
        

        
def search_dicts(dicts: dict, search: str):
    matching_keys = []
    non_matching_keys = []

    for key in dicts.keys():
        if search in key or (key == 'mix' and search in dicts.keys()):
            matching_keys.append(key)
        else:
            non_matching_keys.append(key)

    return matching_keys, non_matching_keys