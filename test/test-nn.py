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
Y = Y.reshape(10, 1)
print(f'Y = {Y}')



# NN layers
alpha = 0.01

def relu(x):
    return (x > 0) * x

def diffRelu(x):
    return x > 0

A = rng.random((2, 5)) * 2 - 1
B = rng.random((5, 1)) * 2 - 1

for i in range(2000):

    error = 0

    if(i % 100 == 0):
        print(f'i = {i}')

    for n in range(len(X)):
        x = X[n]
        y = Y[n]

        h = relu(x @ A)
        output = relu(h @ B)

        error += (output - y) ** 2

        B_delta = (output - y) * diffRelu(h)
        B_delta = B_delta.reshape(5, 1)
        B -= alpha * B_delta

        A_delta = B_delta * diffRelu(x)
        A_delta = A_delta.T

        A -= alpha * A_delta

    if(i % 100 == 0):
        print(f'error = {error}')

print(f'X = {X}')
print(f'Y = {Y}')

h = relu(X @ A)
y = relu(h @ B)
print(f'y = {y}')


