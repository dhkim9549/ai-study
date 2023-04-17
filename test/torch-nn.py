import torch
from torch import nn
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y 

model = NeuralNetwork()
print(f'model = {model}')

# training data
X = np.random.randint(0, 2, size=(20, 2)).astype(float)
print(f'X = {X}')

Y = np.logical_xor(X[:,0], X[:,1]).astype(float)
Y = np.array([Y]).T 
print(f'Y = {Y}')

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

# train
for i in range(1000):

    j = i % len(X)

    #x = X[j:j+1]
    x = X
    x = torch.Tensor(x)

    y = model(x)

    #y0 = Y[j]
    y0 = Y
    y0 = torch.Tensor(y0)
    loss = loss_fn(y, y0)
    if(i % 100 == 0):
        print()
        print(f'i = {i}')
        print(f'loss = {loss}')
        print(f'y = {y}')

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


