import yaml
from pydub import AudioSegment
import numpy as np
import wave
import os
import librosa as librosa
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import glob
import time

start_time = time.time()
#directory = "/Users/odysseaslazaridis/Documents/GroupProject/new_babyslack"
directory = "new_babyslack"
# Size of the Fast Fourier Transform (FFT), which will also be used as the window length
n_fft=1024 #number is from  mdpi paper

# Step or stride between windows. If the step is smaller than the window length, the windows will overlap
hop_length=512

# Specify the window type for FFT/STFT
window_type ='hann'

mel_bins = 40 # Number of mel bands
fmin = 0
fmax= None

def get_iterations(track_path,step_size = 64000):
    inst = os.listdir(track_path)[0]
    instr_path = os.path.join(track_path, inst)
    y, sr = sf.read(instr_path)
    return int(np.floor(y.shape[0]/step_size))


tracks_dict={}
for tr in os.listdir(directory):
    track_lables_dict={}
    track_path = os.path.join(directory, tr)

    file_path= os.path.join(track_path, tr+".pt")
    if os.path.exists(file_path):
        os.remove(file_path)

    if tr!=".DS_Store":
        
        iters = get_iterations(track_path)

        inst_list = ["Guitar.wav","Piano.wav","Bass.wav","Drums.wav"]
        for inst in inst_list:
            if os.path.exists(os.path.join(track_path, inst)):
                labels = []
                instr_path = os.path.join(track_path, inst)
                y, sr = sf.read(instr_path)
                Mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window_type, n_mels = mel_bins, power=2.0)
                mel_spectrogram_db = librosa.power_to_db(Mel_spectrogram, ref=np.max)
                for i in range(iters):
                    start_index = i*int(2*64000/n_fft)
                    end_index = (i+1)*int(2*64000/n_fft)
                    if np.amax(mel_spectrogram_db[:,start_index:end_index]) < -60:
                        labels.append(0)
                    else:
                        labels.append(1)
            else:
                print("this insttrument ("+inst+") doesn't exist")                     
                labels = [0 for i in range(iters)]
            track_lables_dict[inst]=labels

    torch.save(track_lables_dict, os.path.join(track_path, tr + '.pt'))

end_time = time.time()

print("the code run in "+str(end_time-start_time))
