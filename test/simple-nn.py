import numpy as np

def relu(x):
    return x * (x > 0)

def dRelu(output):
    return output>0

X = np.random.randint(0, 2, size=(20, 2)).astype(float)
print(f'X = {X}')

Y = np.logical_xor(X[:,0], X[:,1]).astype(float)
Y = np.array([Y]).T 
print(f'Y = {Y}')

alpha = 0.002

hidden_size = 20

W01 = np.random.normal(0, 1, (2, hidden_size))
W12 = np.random.normal(0, 1, (hidden_size, 1))

for i in range(1000):

    j = i % len(X)

    #x = X[j:j+1]
    x = X 

    # infer
    L1a = x @ W01

    L1 = relu(L1a)

    L2 = L1 @ W12

    #y0 = Y[j]
    y0 = Y 

    d = L2 - y0

    error = sum(d ** 2)
    if(i % 100 == 0):
        print()
        print(f'i = {i}')
        print(f'error = {error}')
        print(f'L2 = {L2}')

    # backpropagation
    dW12 = L1.T @ d
    W12 -= alpha * dW12

    dL1 = d @ W12.T

    dL1a = dL1 * dRelu(L1a)

    dW01 = x.T @ dL1a
    W01 -= alpha * dW01








