import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

C = 3
L = 130
N = 100

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
criterion = nn.MSELoss()

input = torch.randn(N, (2*C+1)*L, dtype=torch.complex128)
target = torch.randn(N, L, dtype=torch.complex128)

optimizer = optim.Adam(dnn.parameters(), lr=0.001)

epochs = 100

for _ in range(epochs):
    optimizer.zero_grad()
    output = dnn(input)
    loss = criterion(output.real, target.real) + criterion(output.imag, target.imag)
    loss.backward()
    optimizer.step()