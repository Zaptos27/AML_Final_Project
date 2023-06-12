import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_generator as dg
from datetime import datetime

#print time
print(datetime.now().strftime("%H:%M:%S"))

C = 3
L = 129
amount = 1000
files = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.seed = 0

class DNN(nn.Module):

    def __init__(self, C, L):
        super(DNN, self).__init__()

        self.layer1 = nn.Linear((2*C+1)*L, L, dtype=torch.float64)
        self.layer2 = nn.Linear(L, L, dtype=torch.float64)
        self.layer3 = nn.Linear(L, L, dtype=torch.float64)
        self.layer4 = nn.Linear(L, L, dtype=torch.float64)
        self.layer5 = nn.Linear(L, L, dtype=torch.float64)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = self.layer5(x)
        return x

dnn = DNN(C, L).to(device)
criterion = nn.MSELoss(reduction='sum')


inst = 'Piano'
data, label = [], []
for f in dg.data_frame(files, amount, C = C, L = L, mix_amount = 4):
    positive, negative = dg.search_dicts(f, inst)
    if inst in positive:
        # Yield positive with inst as label and negative with a zero_like as label
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
N = (len(data)*amount)
print(N)
data = torch.stack(data).to(device)
label = torch.stack(label).to(device)
data = data.reshape(-1, L*(2*C+1))
label = label.reshape(-1, L)
        
#print time
print(datetime.now().strftime("%H:%M:%S"))


input = data.real
target = label.real

# We initialize the first layer with random weights, and optimize them
first_opt = optim.LBFGS(dnn.layer1.parameters(), max_iter=6000)

def closure():
    first_opt.zero_grad()
    output = dnn.layer1(input)
    loss = criterion(output, target)
    loss.backward()
    return loss
first_opt.step(closure)

#print time
print(datetime.now().strftime("%H:%M:%S"))

sp = target
xp = dnn.layer1(input)

sp_mean = sp.mean(0)

#Initializes each layers weights and optimizes them
for layer in [dnn.layer2, dnn.layer3, dnn.layer4, dnn.layer5]:
    xp_mean = xp.mean(0)
    Csx = torch.zeros((L,L),dtype=torch.float64).to(device)
    Cxx = torch.zeros((L,L),dtype=torch.float64).to(device)
    for i in range(N):
        Csx += torch.outer((sp[i]-sp_mean), (xp[i]-xp_mean))
        Cxx += torch.outer((xp[i]-xp_mean), (xp[i]-xp_mean))
    weights = torch.matmul(Csx,torch.inverse(Cxx))
    bias = sp_mean - torch.matmul(weights,xp_mean)
    layer.weight.data.copy_(weights)
    layer.bias.data.copy_(bias)

    layer_opt = optim.LBFGS(layer.parameters(), max_iter=6000)
    def closure():
        layer_opt.zero_grad()
        output = layer(xp)
        loss = criterion(output, target)
        loss.backward(retain_graph=True)
        return loss
    layer_opt.step(closure)
    xp = layer(xp)

# This is a final fine tuning, takes way longer than everything else, dunno if it's necessary
final_opt = optim.LBFGS(dnn.parameters(), max_iter=6000)
def closure():
    final_opt.zero_grad()
    output = dnn(input)
    loss = criterion(output, target)
    loss.backward()
    return loss
final_opt.step(closure)


torch.save(dnn.state_dict(), 'DNN_leastSquares_real.pt')
#print time
print(datetime.now().strftime("%H:%M:%S"))


input = data.imag
target = label.imag
dnn = DNN(C, L).to(device)
# We initialize the first layer with random weights, and optimize them
first_opt = optim.LBFGS(dnn.layer1.parameters(), max_iter=6000)

def closure():
    first_opt.zero_grad()
    output = dnn.layer1(input)
    loss = criterion(output, target)
    loss.backward()
    return loss
first_opt.step(closure)

sp = target
xp = dnn.layer1(input)

sp_mean = sp.mean(0)

#Initializes each layers weights and optimizes them
for layer in [dnn.layer2, dnn.layer3, dnn.layer4, dnn.layer5]:
    xp_mean = xp.mean(0)
    Csx = torch.zeros((L,L),dtype=torch.float64).to(device)
    Cxx = torch.zeros((L,L),dtype=torch.float64).to(device)
    for i in range(N):
        Csx += torch.outer((sp[i]-sp_mean), (xp[i]-xp_mean))
        Cxx += torch.outer((xp[i]-xp_mean), (xp[i]-xp_mean))
    weights = torch.matmul(Csx,torch.inverse(Cxx))
    bias = sp_mean - torch.matmul(weights,xp_mean)
    layer.weight.data.copy_(weights)
    layer.bias.data.copy_(bias)

    layer_opt = optim.LBFGS(layer.parameters(), max_iter=6000)
    def closure():
        layer_opt.zero_grad()
        output = layer(xp)
        loss = criterion(output, target)
        loss.backward(retain_graph=True)
        return loss
    layer_opt.step(closure)
    xp = layer(xp)

# This is a final fine tuning, takes way longer than everything else, dunno if it's necessary
final_opt = optim.LBFGS(dnn.parameters(), max_iter=6000)
def closure():
    final_opt.zero_grad()
    output = dnn(input)
    loss = criterion(output, target)
    loss.backward()
    return loss
final_opt.step(closure)


torch.save(dnn.state_dict(), 'DNN_leastSquares_imag.pt')
#print time
print(datetime.now().strftime("%H:%M:%S"))