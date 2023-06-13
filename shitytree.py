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
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

freq = 1000
# Set the logging level to suppress warning messages
logger = logging.getLogger('lightgbm')
logger.setLevel(logging.ERROR)

directory = "/Users/odysseaslazaridis/Documents/GroupProject/new_babyslack"

n_fft=1024 #number is from  mdpi paper

# Step or stride between windows. If the step is smaller than the window length, the windows will overlap
hop_length=512

# Specify the window type for FFT/STFT
window_type ='hann'

mel_bins = 40 # Number of mel bands
fmin = 0
fmax= None

def get_iterations(track_path,step_size = freq):
    instr_path = os.path.join(track_path, "mix.wav")
    y, sr = sf.read(instr_path)
    return int(np.floor(y.shape[0]/step_size))

mels = []
g_target = []
p_target = []
b_target = []
d_target = []

inst_list = [g_target,p_target,b_target,d_target]

for tr in os.listdir(directory):
    if tr !='.DS_Store':
        tr_dir = os.path.join(directory, tr)
        mix_dir = os.path.join(tr_dir, 'mix.wav')
        lab_dir = os.path.join(tr_dir, f'{tr}.pt')
        iters = get_iterations(tr_dir)
        label_dir = os.path.join(directory)
        y, sr = sf.read(mix_dir)
        Mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window_type, n_mels = mel_bins, power=2.0)
        mel_spectrogram_db = librosa.power_to_db(Mel_spectrogram, ref=np.max)
        labels = torch.load(lab_dir)
        for i in range(iters):
            start_index = i*int(2*freq/n_fft)
            end_index = (i+1)*int(2*freq/n_fft)
            mels.append(mel_spectrogram_db[:,start_index:end_index].flatten())
            g_target.append(labels['Guitar.wav'][i])
            b_target.append(labels['Bass.wav'][i])
            p_target.append(labels['Piano.wav'][i])
            d_target.append(labels['Drums.wav'][i])

mels = np.array(mels)
g_target = np.array(g_target)
b_target = np.array(b_target)
d_target = np.array(d_target)
p_target = np.array(p_target)


inst_list = [g_target,p_target,b_target,d_target]

# Prepare your data
# X_train, X_test: Features
# y_train, y_test: Target variable



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
        'verbose': 0
    }
#Guitar
#Best is trial 92 with value: 0.9022730191344549.
#Best Parameters: {'num_leaves': 97, 'learning_rate': 0.09970970888921578, 'feature_fraction': 0.6540934921314316, 'max_depth': 34}
#Best Accuracy: 0.9022730191344549

#Druns
#Best is trial 93 with value: 0.9630794914601258.
#Best Parameters: {'num_leaves': 98, 'learning_rate': 0.09759602231181187, 'feature_fraction': 0.7919651058558588, 'max_depth': 27}
#Best Accuracy: 0.9630794914601258
    # Create a LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)

    # Train the model
    model = lgb.train(params, train_data, num_boost_round=100)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]  # Convert probabilities to class labels

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

X = mels

auc_scores=[]
fpr_list =[]
tpr_list =[]
for inst in inst_list:

    y = b_target
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create a LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)

    # Perform hyperparameter optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=40)

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


# Plot ROC curve
plt.plot(fpr_list[0], tpr_list[0],color='g', label=f'Guitar AUC = {auc_scores[0]:.2f}')
plt.plot(fpr_list[1], tpr_list[1],color='r', label=f'Piano AUC = {auc_scores[1]:.2f}')
plt.plot(fpr_list[2], tpr_list[2],color='b', label=f'Bass AUC = {auc_scores[2]:.2f}')
plt.plot(fpr_list[3], tpr_list[3],color='y', label=f'Drums AUC = {auc_scores[3]:.2f}')
  
# Naming the x-axis, y-axis and the whole graph
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')


  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()
