import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import os
import numpy as np
import data_generator as dg
from scipy.signal import istft
import soundfile as sf


C = 3
L = 1025
N = 100
L2 = 44100//5
overlap = 10 
inst = 'Drums'

torch.seed = 0

def complex_relu(complex_input):
    return torch.view_as_complex(torch.clamp(torch.view_as_real(complex_input),min = 0))

class DNN(nn.Module):

    def __init__(self, C, L):
        super(DNN, self).__init__()

        self.layer1 = nn.Linear((2*C+1)*L, L, dtype=torch.complex128)
        self.layer2 = nn.Linear(L, L, dtype=torch.complex128)
        self.layer3 = nn.Linear(L, L, dtype=torch.complex128)
        self.layer4 = nn.Linear(L, L, dtype=torch.complex128)
        self.layer5 = nn.Linear(L, L, dtype=torch.complex128)

    def forward(self, x):
        x = complex_relu(self.layer1(x))
        x = complex_relu(self.layer2(x))
        x = complex_relu(self.layer3(x))
        x = complex_relu(self.layer4(x))
        x = self.layer5(x)
        return x

dnn = DNN(C, L)

class DNN2(nn.Module):

    def __init__(self, L2):
        super(DNN2, self).__init__()

        self.layer1 = nn.Linear(L2, L2//2, dtype=torch.float64)
        self.layer2 = nn.Linear(L2//2, L2//2, dtype=torch.float64)
        self.layer3 = nn.Linear(L2//2, L2//2, dtype=torch.float64)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

dnn2 = DNN2(L2)


class torchAgent:
    def __init__(self,model, loss_fn, data_path: str = None, valid_path: str = None, test_path: str = None, optimizer = None,L2: int = None, device: str = None, epoch: int = 100, model_path = None, verbose: int = 2, track_amount: int = None, C: int = C, L: int = L, N: int = N, **kwargs):
        # device: cpu / gpu
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu" # set device
            )
        else:
            self.device = device # set device
        self.model = model.to(self.device) # set model
        self.loss_fn = loss_fn # set loss function
        
        self.scheduler = None # set scheduler
        self.epoch = epoch # set epoch
        self.verbose = verbose # set verbose
        if model_path is None:
            self.model_path = f'model_{datetime.now().strftime("%y_%m_%d_%H%M")}' # set model path
        else:
            self.model_path = model_path

        if data_path is None:
            self.data_path  = 'Data/slakh2100_flac_redux/train' #data path
        else:
            self.data_path = data_path

        if test_path is None:
            self.test_path = 'Data/slakh2100_flac_redux/test' #test path
        else:
            self.test_path = test_path
        if valid_path is None:
            self.valid_path = 'Data/slakh2100_flac_redux/validation' #validataion path
        else:
            self.valid_path = valid_path
        
        if track_amount is None:
            self.track_amount = len(os.listdir(self.data_path))
        else:
            self.track_amount = track_amount
        
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
        
        if L2 is None:
            self.L2 = 44100
        else:
            self.L2 = L2
            
        self.C = C
        self.L = L
        self.N = N

    
    def add_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def add_optimizer(self, optimizer, **kwargs):
        self.optimizer = optimizer(self.model.parameters(), **kwargs)

    def add_scheduler(self, scheduler, **kwargs):
        self.scheduler = scheduler(self.optimizer, **kwargs)
    
    
    def tracks(self, validate: bool = False, test: bool = False, **kwargs):
        if validate:
            path = self.valid_path
        elif test:
            path = self.test_path
        else:
            path = self.data_path
        
        for f in dg.data_frame(50, self.N,directory=path, C = self.C, L = self.L, mix_amount = 4):
            positive, negative = dg.search_dicts(f, inst)
            if inst in positive:
                # Yield positive with inst as label and negative with a zero_like as label
                data, label = [], []
                for instrument in positive: 
                    data.append(torch.view_as_complex(f[instrument]))
                    label.append(torch.view_as_complex(f[inst][:, :,self.C]))
                iter = 0
                for instrument in negative:
                    data.append(torch.view_as_complex(f[instrument]))
                    label.append(torch.view_as_complex(torch.zeros_like(f[inst][:, :,self.C])))
                    iter += 1
                    if iter == len(positive):
                        break
                data = torch.stack(data).to(self.device)
                label = torch.stack(label).to(self.device)
                data = data.reshape(-1, self.L*(2*self.C+1)).to(self.device)
                label = label.reshape(-1, self.L).to(self.device)
                yield data, label
                del data, label
                
    def track_original(self, validate: bool = False, test: bool = False, **kwargs):
        if validate:
            path = self.valid_path
        elif test:
            path = self.test_path
        else:
            path = self.data_path
            
        for f in dg.data_dicts(self.N, directory=path, mixing=False, dict1=True, clean=True):
            positive, negative = dg.search_dicts(f, inst)
            if inst in positive:
                la = torch.from_numpy(f[inst])
                dat = []
                labels = []
                for instrument in positive:
                    data = torch.from_numpy(f[instrument])
                    for i in torch.randint(0,data.shape[0]-self.L2-1,(10,)):
                        dat.append(data[i:i+self.L2])
                        labels.append(la[i+self.L2//2:i+self.L2])
                for instrument in negative:
                    data = torch.from_numpy(f[instrument])
                    for i in torch.randint(0,data.shape[0]-self.L2-1,(2,)):
                        dat.append(data[i:i+self.L2])
                        labels.append(torch.zeros_like(la[i+self.L2//2:i+self.L2]))
                dat = torch.stack(dat).to(self.device)
                labels = torch.stack(labels).to(self.device)
                dat = dat.reshape(-1, self.L2).to(self.device)
                labels = labels.reshape(-1, self.L2//2).to(self.device)
                if clean:
                    #set all values to under 1e-5 to zero
                    dat[torch.abs(dat) < 1e-5] = 0
                yield dat, labels
                del data, labels
                if self.device == 'cuda':
                    torch.cuda.empty_cache()   
            
                        

    def train_one_epoch(self, **kwargs):
        self.model.train(True)
        running_loss = 0.

        for i, (data, labels) in enumerate(self.tracks()):
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # calculate loss
            loss = self.loss_fn(1000*torch.view_as_real(self.model(data)), 1000*torch.view_as_real(labels))

            # backpropagation
            loss.backward()

            # update parameters
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()
            print(f'Batch: [{i+1}] loss: {loss.item():.5f}, loss: {running_loss:.5f}',end='\r')

            # free memory
            del data, labels, loss
            torch.cuda.empty_cache()

        self.model.train(False)
        return running_loss/(i+1)
    
    def validate(self, **kwargs):
        self.model.train(False)
        running_loss = 0.

        for i, (data, labels) in enumerate(self.tracks(validate=True)):
            # calculate loss
            loss = self.loss_fn(torch.view_as_real(self.model(data)), torch.view_as_real(labels))

            # print statistics
            running_loss += loss.item()
            print(f'Batch: [{i+1}] loss: {loss.item():.5f}, loss: {running_loss:.5f}',end='\r')

            # free memory
            del data, labels, loss
            torch.cuda.empty_cache()

        return running_loss/(i+1)

    def train(self, **kwargs):
        best_loss = np.inf
        for epoch in range(self.epoch):
            print(f'Epoch: [{epoch+1}/{self.epoch}]')
            epoch_loss = self.train_one_epoch(**kwargs)
            print(f'Epoch: [{epoch+1}/{self.epoch}] loss: {epoch_loss:.5f}')
            valid_loss = self.validate(**kwargs)
            if best_loss > valid_loss:
                print('Saving model...')
                self.save_model(epoch=str(epoch+1))
                best_loss = valid_loss
            if self.scheduler is not None:
                self.scheduler.step()
        print('Finished Training')

    def save_model(self, epoch: str = 'final'):
        torch.save(self.model.state_dict(), self.model_path+'_EPOCH_'+epoch)
        print(f'Model saved at {self.model_path}')

    def load_model(self, model_path: str):
        self.model.load_state_dict(torch.load(model_path))
        print(f'Model loaded from {model_path}')     

    def test(self, **kwargs): #Loss function for test?
        self.model.train(False)
        running_loss = 0.

        for i, (data, labels) in enumerate(self.tracks(test=True)):
            # calculate loss
            loss = self.loss_fn(torch.view_as_real(self.model(data)), torch.view_as_real(labels))

            # print statistics
            running_loss += loss.item()
            print(f'Batch: [{i+1}] loss: {loss.item():.5f}, loss: {running_loss:.5f}',end='\r')

            # free memory
            del data, labels, loss
            torch.cuda.empty_cache()

        return running_loss/(i+1)   
    
    #Predict from the test set
    def predict(self, **kwargs):
        self.model.train(False)
        
        for i, (data, labels) in enumerate(self.tracks(test=True)):
            self.model(data)
            
            yield data, labels

            # free memory
            del data, labels
            torch.cuda.empty_cache()
        
    def generate_track(self, track_amount: int = 1, **kwargs):
        self.model.eval()
        for data in dg.data_dicts(track_amount,directory=self.test_path, print_dict=True):
            new_dat = []
            dat = torch.view_as_complex(data['mix']).to(self.device)
            output = torch.zeros_like(dat).to(self.device)
            output = torch.transpose(output, 0, 1)
            
            for i in range(dat.shape[1]):
                if i < self.C:
                    continue
                elif i > dat.shape[1]-self.C-1:
                    continue
                new_dat.append(dat[: , i-self.C:i+self.C+1])
            
            new_dat = torch.stack(new_dat)
            new_dat = new_dat.reshape(-1, L*(2*C+1)).to(self.device)
            
            output=self.model(new_dat)
            output = torch.transpose(output, 0, 1)
            if track_amount > 1:
                yield istft(output.detach().cpu(), fs = 44100, noverlap=overlap)[1]
        output = output/torch.sum(torch.abs(output))
        yield istft(output.detach().cpu(), fs = 44100, noverlap=overlap)[1]

    def generate_track_from_file(self, file_path: str, **kwargs):
        self.model.eval()
        data = sf.read(file_path)[0]
        new_dat = []
        dat = torch.view_as_complex(torch.from_numpy(data)).to(self.device)

        for i in range(dat.shape[1]):
            if i < self.C:
                continue
            elif i > dat.shape[1]-self.C-1:
                continue
            new_dat.append(dat[: , i-self.C:i+self.C+1])

        new_dat = torch.stack(new_dat)
        new_dat = new_dat.reshape(-1, L*(2*C+1)).to(self.device)

        output=self.model(new_dat)
        output = torch.transpose(output, 0, 1)
        print(output.shape)
        yield istft(output.detach().cpu(), fs = 44100, noverlap=overlap)[1]



if __name__ == '__main__':
    criterion = nn.MSELoss(reduction='sum')#nn.L1Loss(reduction='sum')#nn.MSELoss(reduction='sum')
    epochs = 100

    agent = torchAgent(dnn, criterion, epoch=epochs, L = L, C = C)
    agent.add_optimizer(optim.Adam, lr=0.01)
    agent.add_scheduler(optim.lr_scheduler.StepLR, step_size=2, gamma=0.75)
    agent.load_model('model_23_06_14_1354_EPOCH_35')
    #agent.load_model('model_23_06_12_2340_EPOCH_6')
    #agent.validate()
    #agent.train()
    sf.write('test.wav', list(agent.generate_track(track_amount=1))[0], 44100)
