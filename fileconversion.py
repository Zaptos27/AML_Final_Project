import yaml
from pydub import AudioSegment
import numpy as np
import wave
import os

old_directory = "babyslakh_16k"
new_directory = "new_babyslack"
try:
    os.mkdir(new_directory)
except:
    pass

for tr in os.listdir(old_directory):
    old_tr = os.path.join(old_directory, tr)
    new_tr = os.path.join(new_directory, tr)
    try:
        os.mkdir(new_tr)
    except:
        pass

    with open(os.path.join(old_tr, "metadata.yaml")) as meta:
        metadata = yaml.load(meta, Loader=yaml.Loader)

    file_inst = []
    for stem in metadata['stems'].keys():
        file_inst.append([stem + '.wav', metadata['stems'][stem]['inst_class']])

    file_inst = np.array(file_inst)
    file_inst = file_inst[[file_inst[:,0][i] in os.listdir(old_tr + '/stems') for i in range(len(file_inst))]]
    
    mix_path = os.path.join(new_tr, 'mix.wav')

    mix = AudioSegment.from_file(os.path.join(old_tr, 'mix.wav'))
    mix.export(mix_path, format='wav')

    unique_inst = np.unique(file_inst[:,1])
    for inst in unique_inst:
        inst_list = []
        inst_path = os.path.join(new_tr, inst + '.wav')
        for i in range(file_inst.shape[0]):
            if file_inst[:,1][i] == inst:
                inst_list.append(file_inst[:,0][i])
        if len(inst_list) > 1:
            sound1 = AudioSegment.from_file(os.path.join(old_tr + '/stems', inst_list[0]))
            for i in range(len(inst_list)-1):
                sound2 = AudioSegment.from_file(os.path.join(old_tr + '/stems', inst_list[i+1]))
                combined = sound1.overlay(sound2)
                sound1 = combined
            combined.export(inst_path, format='wav')
        else:
            sound = AudioSegment.from_file(os.path.join(old_tr + '/stems', inst_list[0]))
            sound.export(inst_path, format='wav')