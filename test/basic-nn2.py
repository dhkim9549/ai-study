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

A = np.random.normal(0, 1, (2, 15))
A2 = np.random.normal(0, 1, (15, 15))
B = np.random.normal(0, 1, (15, 1))
alpha = 0.001
print(f'A = {A}')
print(f'A2 = {A2}')
print(f'B = {B}')

for i in range(100000):

    error = 0

    for n in range(len(X)):
        x = np.array([X[n]])
        y = np.array([Y[n]])

        # infer
        h = relu(x @ A)
        h2 = relu(h @ A2)
        output = h2 @ B

        error += (output - y) ** 2

        B -= alpha * (output - y) * h2.T 
        A2 -= alpha * (output - y) * B.T * diffRelu(h @ A2) * h.T 
        Ax = alpha * (output - y) * B.T * diffRelu(h @ A2) * A2.T * diffRelu(x @ A) @ A.T
        A -= Ax.T

    if(i % 100 == 0):
        print()
        print(f'i = {i}')
        print(f'error = {error}')

