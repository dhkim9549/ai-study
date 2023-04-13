########################################################
#  simple neural network using pytorch backward function
########################################################

import numpy as np
import torch

# training data 
X = np.random.randint(2, size=(40, 2))
X = X.astype(float)
print(f'X = {X}')

Y = np.logical_xor(X[:,0], X[:,1])
Y = Y.astype(float)
Y = np.array([Y]).T 
print(f'Y = {Y}')

A = np.random.normal(0, 1, (2, 10))
B = np.random.normal(0, 1, (10, 1))
A = torch.tensor(A, requires_grad=True)
B = torch.tensor(B, requires_grad=True)

alpha = 0.001

for i in range(10000):

    error = 0

    for n in range(len(X)):
        x = torch.tensor(X[n], requires_grad=True)
        y = torch.tensor(Y[n])

        # infer
        h = torch.relu(x @ A)
        output = h @ B

        error = (output - y) ** 2

        # backpropagation
        error.backward(retain_graph=True)

        # update weights
        with torch.no_grad():
            B = B - alpha * B.grad
            B = B.clone().detach().requires_grad_(True)
            A = A - alpha * A.grad
            A = A.clone().detach().requires_grad_(True)

        x.grad.zero_()

    if(i % 1000 == 0):
        print()
        print(f'i = {i}')
        print(f'error = {error}')
        print(f'B = {B}')
        print(f'A = {A}')

