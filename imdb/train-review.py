##################################################
# IMDB review pos/neg prediction
##################################################

from os import listdir
from os.path import isfile, join
import numpy as np

# Load vocab
f = open("/root/data/aclImdb/imdb.vocab", "r")
i = 0
voca = {} 
voca2 = {}
for x in f:
    token = x.split("\n")[0]
    voca[token] = i
    voca2[i] = token
    i += 1
    if(i >= 10000):
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

W01 = np.random.normal(0, 1, (len(voca), hidden_size))
W12 = np.random.normal(0, 1, (hidden_size, 2))

# train
fileLst = []

pathPos = '/root/data/aclImdb/train/pos'
filesPos = [f for f in listdir(pathPos) if isfile(join(pathPos, f))]
for fileNm in filesPos:
    fileLst.append(pathPos + '/' + fileNm)

pathNeg = '/root/data/aclImdb/train/neg'
filesNeg = [f for f in listdir(pathNeg) if isfile(join(pathNeg, f))]
for fileNm in filesNeg:
    fileLst.append(pathNeg + '/' + fileNm)

crctCnt = 0
totCnt = 0
for cnt in range(4000000):
    reviewFile = fileLst[np.random.randint(len(fileLst))]

    f = open(reviewFile, "r")
    tokenLst = []
    for line in f:
        tokenLst += line.replace(".", "").replace("!", "").lower().split(" ")

    x = np.zeros((1, len(voca)), dtype=float)

    for token in tokenLst:
        if token in voca:
            i = voca[token]
            x[0, i] += 1
    x = np.minimum(x, 1.0)

    y0 = np.array([[1.0, 0.0]])
    if "neg" in reviewFile:
        y0 = np.array([[0.0, 1.0]])

    # infer
    L1a = x @ W01
    L1 = relu(L1a)
    L2 = softmax(L1 @ W12)
    y = L2

    d = L2 - y0
    error = (d ** 2).sum()

    # stat
    totCnt += 1
    if (y[0, 0] > 0.5 and y0[0, 0] > 0.5) or (y[0, 0] <  0.5 and y0[0, 0] <  0.5):
        crctCnt += 1
    crctRat = crctCnt / totCnt 
    if cnt % 1000 == 0:
        print()
        print(f'cnt = {cnt}')
        print(f'crctRat = {crctRat}')
        print(f'error = {error}')
        print(f'y = {y}')
        print(f'y0 = {y0}')
        totCnt = 0
        crctCnt = 0

    # backpropagation
    dW12 = L1.T @ d
    W12 -= alpha * dW12

    dL1 = d @ W12.T

    dL1a = dL1 * dRelu(L1a)

    dW01 = x.T @ dL1a
    W01 -= alpha * dW01


