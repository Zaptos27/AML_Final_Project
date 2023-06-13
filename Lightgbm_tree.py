import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import librosa
import scipy.signal as si
import os
import soundfile as sf
import torch
import optuna
import logging
from sklearn.metrics import roc_curve, roc_auc_score
import catboost as cat
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
import data_generator as dg

freq = 1000
# Set the logging level to suppress warning messages
logger = logging.getLogger('lightgbm')
logger.setLevel(logging.ERROR)

directory = "dara"

n_fft=2048 #number is from  mdpi paper

# Step or stride between windows. If the step is smaller than the window length, the windows will overlap
hop_length=256

# Specify the window type for FFT/STFT
window_type ='hann'

sr = 44100

sec = 2

frames = (sec*sr//hop_length)

mel_bins = 40 # Number of mel bands
fmin = 0
fmax= None

tracks = np.empty((0,40*frames))
track_labels = {}
inst_list = ['Bass', 'Chromatic Percussion', 'Drums', 'Guitar', 'Piano', 'Strings (continued)', 'Organ', 'Synth Pad', 'Brass', 'Reed']

for f in dg.data_dicts(3,directory='dara', sample_freq=sr, mixing=False,dict1=True, print_dict=True):   
    for inst in f.keys():
        label = []
        if inst == 'mix':
            continue
        mel = librosa.feature.melspectrogram(y=f[inst], sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window_type, n_mels = mel_bins, power=2.0)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        lenght = mel_db.shape[1]
        num_chunks = lenght // frames
        for db in np.hsplit(mel_db[:, :num_chunks*frames], num_chunks):
            if np.max(db) < -60:
                label.append(0)
            else:
                label.append(1)
        if inst not in track_labels.keys():    
            track_labels[inst] = label
        else:
            track_labels[inst] = np.concatenate((track_labels[inst],label))
    for inst in inst_list:
        if inst not in f.keys():
            label = []
            for i in range(num_chunks):
                label.append(0)
            if inst not in track_labels.keys():    
                track_labels[inst] = label
            else:
                track_labels[inst] = np.concatenate((track_labels[inst],label))
    
    mel = librosa.feature.melspectrogram(y=f['mix'], sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window_type, n_mels = mel_bins, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    for db in np.hsplit(mel_db[:, :num_chunks*frames], num_chunks):
        tracks = np.vstack((tracks,db.flatten()))
    
# Set the hyperparameters for the model
def objective(trial):
    # Set the hyperparameters for the model
    param = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.5),
        'depth': trial.suggest_int('depth', 4, 10),
    }
    
    # Create a CatBoost classifier with the suggested hyperparameters
    model = cat.CatBoostClassifier(task_type='GPU', **param)
#Guitar
#Best is trial 92 with value: 0.9022730191344549.
#Best Parameters: {'num_leaves': 97, 'learning_rate': 0.09970970888921578, 'feature_fraction': 0.6540934921314316, 'max_depth': 34}
#Best Accuracy: 0.9022730191344549

#Druns
#Best is trial 93 with value: 0.9630794914601258.
#Best Parameters: {'num_leaves': 98, 'learning_rate': 0.09759602231181187, 'feature_fraction': 0.7919651058558588, 'max_depth': 27}
#Best Accuracy: 0.9630794914601258
    # Create a LightGBM dataset
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]  # Convert probabilities to class labels

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy