########################################################
#  simple neural network using only numpy 
########################################################

import numpy as np


# training data 
rng = np.random.default_rng()
X = rng.integers(2, size=(20, 2))
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

A = np.random.normal(0, 1, (2, 50)) * 2 - 1
A0 = np.array(A)
B = np.random.normal(0, 1, (50, 1)) * 2 - 1
B0 = np.array(B)
alpha = 0.001


# train NN
for i in range(10000):

    error = 0

    if(i % 1000 == 1):
        print()
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
        B -= alpha * h.T @ B_delta

        A_delta = B_delta * B.T * diffRelu(x @ A)
        A -= alpha * x.T @ A_delta

        if(i % 1000 == 1):
            print()
            print(f'x = {x}')
            print(f'h = {h}')
            print(f'y = {y}')
            print(f'output = {output}')
            print(f'A = {A}')
            print(f'B = {B}')
            print(f'B_delta = {B_delta}')
            print(f'A_delta = {A_delta}')
            print(f'error = {error}')

print(f'A0 = {A0}')
print(f'B0 = {B0}')



