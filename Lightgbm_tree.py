import numpy as np
import librosa
import optuna
import logging
from sklearn.metrics import roc_curve, roc_auc_score
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import data_generator as dg

freq = 1000
# Set the logging level to suppress warning messages
logger = logging.getLogger('lightgbm')
logger.setLevel(logging.ERROR)

directory = "dara"

n_fft=1024 #number is from  mdpi paper

# Step or stride between windows. If the step is smaller than the window length, the windows will overlap
hop_length=512

# Specify the window type for FFT/STFT
window_type ='hann'

sr = 44100

sec = 1

frames = (sec*sr//hop_length)

mel_bins = 40 # Number of mel bands
fmin = 0
fmax= None


inst_list = ['Guitar', 'Piano','Bass', 'Drums',  'Strings (continued)']
def gener(N,test = False):
    tracks = np.empty((0,mel_bins*frames))
    track_labels = {}
    if test:
        pa = 'Data/slakh2100_flac_redux/test'
    else:
        pa = 'Data/slakh2100_flac_redux/train'
    for f in dg.data_dicts(N,directory=pa, sample_freq=sr, mixing=False,dict1=True, print_dict=True, inst_list=inst_list):   
        for inst in f.keys():
            label = []
            if inst == 'mix' or inst not in inst_list:
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
    return tracks, track_labels

     

# Set the hyperparameters for the model
def objective(trial):
    # Set the hyperparameters for the model
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 30, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
        'max_depth': trial.suggest_int('max_depth', 5, 40),
        'verbose': -1
    }
#Guitar
#Best is trial 92 with value: 0.9022730191344549.
#Best Parameters: {'num_leaves': 97, 'learning_rate': 0.09970970888921578, 'feature_fraction': 0.6540934921314316, 'max_depth': 34}
#Best Accuracy: 0.9022730191344549

#Druns
#Best is trial 93 with value: 0.9630794914601258.
#Best Parameters: {'num_leaves': 98, 'learning_rate': 0.09759602231181187, 'feature_fraction': 0.7919651058558588, 'max_depth': 27}
#Best Accuracy: 0.9630794914601258
    train_data = lgb.Dataset(X_train, label=y_train)
    # Train the model
    model = lgb.train(params, train_data, num_boost_round=100)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]  # Convert probabilities to class labels

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy



auc_scores=[]
fpr_list =[]
tpr_list =[]
X_train, y_train_dict = gener(150)
X_test, y_test_dict = gener(20, test=True)
for inst in inst_list:
    y_train = y_train_dict[inst]
    y_test = y_test_dict[inst]

    train_data = lgb.Dataset(X_train, label=y_train)

    # Perform hyperparameter optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Get the best hyperparameters and corresponding accuracy
    best_params = study.best_params
    best_accuracy = study.best_value
    print("Best Parameters:", best_params)
    print("Best Accuracy:", best_accuracy)

    best_model = lgb.train(best_params, train_data, num_boost_round=100)
    y_pred = best_model.predict(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    # Compute AUC score
    auc_scores.append(roc_auc_score(y_test, y_pred))
    
    print("AUC Score:", auc_scores[-1])

# Plot ROC curve
plt.plot(fpr_list[0], tpr_list[0],color='g', label=f'Guitar AUC = {auc_scores[0]:.2f}')
plt.plot(fpr_list[1], tpr_list[1],color='r', label=f'Piano AUC = {auc_scores[1]:.2f}')
plt.plot(fpr_list[2], tpr_list[2],color='b', label=f'Bass AUC = {auc_scores[2]:.2f}')
plt.plot(fpr_list[3], tpr_list[3],color='y', label=f'Drums AUC = {auc_scores[3]:.2f}')
plt.plot(fpr_list[4], tpr_list[4],color='k', label=f'Strings AUC = {auc_scores[4]:.2f}')
  
# Naming the x-axis, y-axis and the whole graph
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')


  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
plt.savefig('ROC.pdf')
# To load the display window
import pickle
with open('auc_scores.pkl', 'wb') as f:
    pickle.dump(auc_scores, f)
    
with open('fpr_list.pkl', 'wb') as f:
    pickle.dump(fpr_list, f)
    
with open('tpr_list.pkl', 'wb') as f:
    pickle.dump(tpr_list, f)
plt.show()
