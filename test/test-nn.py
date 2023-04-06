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

def reluDiff(x):
    return x > 0

A = rng.random((2, 5)) * 2 - 1
B = rng.random((5, 1)) * 2 - 1
B = B
alpha = 0.01


# train NN
for i in range(10000):

    error = 0

    if(i % 1000 == 1):
        print(f'i = {i}')

    for n in range(len(X)):
        x = X[n]
        y = Y[n]

        # infer
        h = relu(x @ A)
        output = h @ B

        # calculate error
        error += (output - y) ** 2

        # backpropagation
        B_delta = output - y
        B -= alpha * B_delta * np.array([h]).T

        A_delta = B_delta * np.array([reluDiff(x)]).T * B.T
        A -= alpha * A_delta

    if(i % 1000 == 1):
        print(f'error = {error}')

print(f'X = {X}')
print(f'Y = {Y}')

h = relu(X @ A)
y = relu(h @ B)
print(f'y = {y}')

print(f'A = {A}')
print(f'B = {B}')

