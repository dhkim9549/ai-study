import numpy as np

def relu(x):
    return x * (x > 0)

def dRelu(output):
    return output>0

streetlights = np.array( [[ 1, 0, 1 ],
                          [ 0, 1, 1 ],
                          [ 0, 0, 1 ],
                          [ 1, 1, 1 ] ] )

walk_vs_stop = np.array([[ 1, 1, 0, 0]]).T

alpha = 0.002

hidden_size = 40

W01 = np.random.normal(0, 1, (3,hidden_size))
W12 = np.random.normal(0, 1, (hidden_size,1))

print(f'streetlights = {streetlights}')

for i in range(1000):

    j = i % len(streetlights)

    #x = streetlights[j:j+1]
    x = streetlights

    # infer
    L1a = x @ W01

    L1 = relu(L1a)

    L2 = L1 @ W12

    #y0 = walk_vs_stop[j]
    y0 = walk_vs_stop

    d = L2 - y0

    error = sum(d ** 2)
    if(i % 100 == 0):
        print(f'error = {error}')
        print(f'L2 = {L2}')

    # backpropagation
    dW12 = L1.T @ d
    W12 -= alpha * dW12

    dL1 = d @ W12.T

    dL1a = dL1 * dRelu(L1a)

    dW01 = x.T @ dL1a
    W01 -= alpha * dW01








