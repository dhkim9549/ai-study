########################################################
#  simple neural network using only numpy 
########################################################

import numpy as np


# training data 
X = np.random.randint(2, size=(50, 2))
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

A = np.random.normal(0, 1, (2, 50))
B = np.random.normal(0, 1, (50, 1))
alpha = 0.01
print(f'A = {A}')
print(f'B = {B}')

for i in range(10):

    error = 0

    for n in range(len(X)):
        x = np.array([X[n]])
        y = np.array([Y[n]])

        # infer
        h = relu(x @ A)
        output = h @ B

        error += (output - y) ** 2

        B -= alpha * (output - y) * h.T 
        A -= alpha * (output - y) * B.T * diffRelu(x @ A) * x.T 

    if(i % 1 == 0):
        print()
        print(f'i = {i}')
        print(f'error = {error}')

