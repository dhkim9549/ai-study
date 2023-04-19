from os import listdir
from os.path import isfile, join
import numpy as np

# Load vocab
f = open("/root/data/aclImdb/imdb.vocab", "r")
i = 0
voca = {} 
voca2 = {}
for x in f:
    i += 1
    token = x.split("\n")[0]
    voca[token] = i
    voca2[i] = token
    if i > 1000:
        break

# nn
def relu(x):
    return x * (x > 0)

def dRelu(output):
    return output>0

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

alpha = 0.002
hidden_size = 100 

W01 = np.random.normal(0, 1, (1000, hidden_size))
W12 = np.random.normal(0, 1, (hidden_size, 2))

# train
mypath = '/root/data/aclImdb/train/pos'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for cnt in range(40):
    reviewFile = onlyfiles[0]

    f = open(mypath + '/' + reviewFile, "r")
    tokenLst = []
    for line in f:
        tokenLst += line.replace(".", "").replace("!", "").lower().split(" ")

    x = np.zeros((1, 1000), dtype=float)

    for token in tokenLst:
        if token in voca:
            i = voca[token]
            x[0, i] += 1
    x = np.minimum(x, 1.0)

    y0 = np.array([[1.0, 0.0]])

    # infer
    L1a = x @ W01
    L1 = relu(L1a)
    L2 = softmax(L1 @ W12)

    d = L2 - y0
    error = (d ** 2).sum()
    print(f'error = {error}')

    # backpropagation
    dW12 = L1.T @ d
    W12 -= alpha * dW12

    dL1 = d @ W12.T

    dL1a = dL1 * dRelu(L1a)

    dW01 = x.T @ dL1a
    W01 -= alpha * dW01


