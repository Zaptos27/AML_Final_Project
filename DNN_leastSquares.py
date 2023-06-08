import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

C = 3
L = 130
N = 100

torch.seed = 0

class DNN(nn.Module):

    def __init__(self, C, L):
        super(DNN, self).__init__()

        self.layer1 = nn.Linear((2*C+1)*L, L)
        self.layer2 = nn.Linear(L, L)
        self.layer3 = nn.Linear(L, L)
        self.layer4 = nn.Linear(L, L)
        self.layer5 = nn.Linear(L, L)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = self.layer5(x)
        return x

dnn = DNN(C, L)
criterion = nn.MSELoss()

input = torch.randn(N, (2*C+1)*L)
target = torch.randn(N, L)

# We initialize the first layer with random weights, and optimize them
first_opt = optim.LBFGS(dnn.layer1.parameters(), max_iter=600)

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
    Csx = torch.zeros((L,L))
    Cxx = torch.zeros((L,L))
    for i in range(N):
        Csx += torch.outer((sp[i]-sp_mean), (xp[i]-xp_mean))
        Cxx += torch.outer((xp[i]-xp_mean), (xp[i]-xp_mean))
    weights = torch.matmul(Csx,torch.inverse(Cxx))
    bias = sp_mean - torch.matmul(weights,xp_mean)
    layer.weight.data.copy_(weights)
    layer.bias.data.copy_(bias)

    layer_opt = optim.LBFGS(layer.parameters(), max_iter=600)
    def closure():
        layer_opt.zero_grad()
        output = layer(xp)
        loss = criterion(output, target)
        loss.backward(retain_graph=True)
        return loss
    layer_opt.step(closure)
    xp = layer(xp)

# This is a final fine tuning, takes way longer than everything else, dunno if it's necessary
final_opt = optim.LBFGS(dnn.parameters(), max_iter=3000)
def closure():
    final_opt.zero_grad()
    output = dnn(input)
    loss = criterion(output, target)
    loss.backward()
    return loss
final_opt.step(closure)