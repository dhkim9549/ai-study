########################################################
#  simple neural network using only numpy 
########################################################

import numpy as np


# training data 
rng = np.random.default_rng()
X = rng.integers(2, size=(10, 2))
X = X.astype(float)
print(f'X = {X}')

Y = np.logical_xor(X[:,0], X[:,1])
Y = Y.astype(float)
Y = np.array([Y]).T 
print(f'Y = {Y}')


# NN layers
def relu(x):
    return (x > 0) * x

def diffRelu(x):
    return x > 0

A = rng.random((2, 5)) * 2 - 1
B = rng.random((5, 1)) * 2 - 1
alpha = 0.1


# train NN
for i in range(100):

    error = 0

    if(i % 10 == 1):
        print(f'i = {i}')

    for n in range(len(X)):
        x = np.array([X[n]])
        y = np.array([Y[n]])

        # infer
        h = relu(x @ A)
        output = h @ B

        # calculate error
        error += (output - y) ** 2

        # backpropagation
        B_delta = output - y
        B -= alpha * B_delta * h.T

        A_delta = B_delta * diffRelu(h) * B.T
        A -= alpha * x.T @ A_delta

    if(i % 10 == 1):
        print(f'x = {x}')
        print(f'h = {h}')
        print(f'y = {y}')
        print(f'A = {A}')
        print(f'B = {B}')
        print(f'error = {error}')

