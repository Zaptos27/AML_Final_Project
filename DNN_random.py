import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import os
import numpy as np
import data_generator as dg

dropout = False
C = 3
L = 129
N = 1000

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
        if dropout:
            x = F.dropout(x, p=0.2)
        x = complex_relu(self.layer2(x))
        x = complex_relu(self.layer3(x))
        if dropout:
            x = F.dropout(x, p=0.2)
        x = complex_relu(self.layer4(x))
        x = self.layer5(x)
        return x

dnn = DNN(C, L)
criterion = nn.MSELoss(reduction='sum')

optimizer = optim.Adam(dnn.parameters(), lr=0.001)

epochs = 100


class torchAgent:
    def __init__(self,model, loss_fn, data_path: str = None, valid_path: str = None, test_path: str = None, optimizer = None, device: str = None, epoch: int = 100, model_path = None, verbose: int = 2, track_amount: int = None, C: int = C, L: int = L, N: int = N, **kwargs):
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

        if data_path is not None:
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
        
        
        self.optimizer = optimizer(self.model.parameters(), **kwargs) # set optimizer
        
        self.C = C
        self.L = L
        self.N = N

    
    def add_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def add_optimizer(self, optimizer, **kwargs):
        self.optimizer = optimizer(self.model.parameters(), **kwargs)

    def add_scheduler(self, scheduler, **kwargs):
        self.scheduler = scheduler(self.optimizer, **kwargs)
    
    
    def tracks(self, validate: bool = False, test: bool = False):
        if validate:
            path = self.valid_path
        elif test:
            path = self.test_path
        else:
            path = self.data_path
        inst = 'Piano'
        for f in dg.data_frame(50, self.N, C = self.C, L = self.L, mix_amount = 4):
            positive, negative = dg.search_dicts(f, inst)
            if inst in positive:
                # Yield positive with inst as label and negative with a zero_like as label
                data, label = [], []
                for instrument in positive: 
                    data.append(torch.view_as_complex(f[instrument]))
                    label.append(torch.view_as_complex(f[inst][:, :,C]))
                iter = 0
                for instrument in negative:
                    data.append(torch.view_as_complex(f[instrument]))
                    label.append(torch.view_as_complex(torch.zeros_like(f[inst][:, :,C])))
                    iter += 1
                    if iter == len(positive):
                        break
                data = torch.stack(data).to(self.device)
                label = torch.stack(label).to(self.device)
                data = data.reshape(-1, L*(2*C+1))
                label = label.reshape(-1, L)
                yield data, label
                del data, label

    def train_one_epoch(self, **kwargs):
        self.model.train(True)
        running_loss = 0.

        for i, (data, labels) in enumerate(self.tracks()):
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # calculate loss
            loss = self.loss_fn(torch.view_as_real(self.model(data)), torch.view_as_real(labels))

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
        
        
agent = torchAgent(dnn, criterion, optimizer=optim.Adam, epoch=epochs)
agent.add_scheduler(optim.lr_scheduler.StepLR, step_size=3, gamma=0.5)
agent.train()
