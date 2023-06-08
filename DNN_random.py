import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

C = 3
L = 100

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

input = torch.randn(100, (2*C+1)*L)
target = torch.randn(100, L)

optimizer = optim.LBFGS(dnn.parameters(), max_iter=6000)

def closure():
    optimizer.zero_grad()
    output = dnn(input)
    loss = criterion(output, target)
    loss.backward()
    return loss
optimizer.step(closure)
print(dnn(input)[0])
print(target[0])