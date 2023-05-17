##################################################
# Train VOC data
##################################################

import re
import numpy as np
import torch
from torch import nn

vocDict = {}
brcdMap = {}

# Make vocab
f = open("/data/v-xy19-21/20230406_VOC_XY_DATA_200000-210000.txt", "r")
i = 0
for x in f:
    i += 1
    if i == 1:
        continue

    x = x.replace('&#039;', ' ').replace('\n', ' ').replace('&quot;', ' ')
    x = re.sub(r'[:\*\?/\(\)\[\]~\.,\\？!]', ' ', x)

    tokens = x.split("\t")
    brcd = tokens[8]
    if '지사' in brcd:
        brcd = '지사'

    cont = tokens[3] + ' ' + tokens[4]

    if brcd in brcdMap:
        brcdMap[brcd] += 1
    else:
        brcdMap[brcd] = 1
    if brcd in vocDict:
        dataLst = vocDict[brcd]
        dataLst.append(cont)
    else:
        dataLst = []
        dataLst.append(cont)
        vocDict[brcd] = dataLst

    if i > 4000000000:
        break

x = brcdMap
x = {k: v for k, v in sorted(x.items(), key=lambda item: - item[1])}

vocDict = dict(filter(lambda elem:len(elem[1]) > 1, vocDict.items()))

print(brcdMap)
print(list(vocDict.keys()))

brcdLst = list(vocDict.keys())
print(f'brcdLst = {brcdLst}')
print(f'len(brcdLst) = {len(brcdLst)}')

# Load vocab
f = open("voc-vocab.txt", "r")
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

# Cenverts str to numpy array
def strToVec(str):
    tokenLst = []
    tokenLst += str.replace(".", "").replace("!", "").lower().split(" ")

    x = np.zeros((1, len(voca)), dtype=float)

    for token in tokenLst:
        if token in voca:
            i = voca[token]
            x[0, i] += 1
    x = np.minimum(x, 1.0)
    x = torch.Tensor(x)

    return x

# nn
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(len(voca), 100),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(100, len(brcdLst)),
            nn.Softmax()
        )

    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y 

model = NeuralNetwork()
print(f'model = {model}')
print()

loss_fn = nn.CrossEntropyLoss()
print(f'loss_fn = {loss_fn}')
print()
optimizer = torch.optim.SGD(model.parameters(), lr=0.002)
print(f'optimizer = {optimizer}')
print()

crctCnt = 0
totCnt = 0
for cnt in range(2000000000000000000000000000):
    brcd = brcdLst[np.random.randint(len(brcdLst))]

    contLst = vocDict[brcd]

    cont = contLst[np.random.randint(len(contLst))]

    x = strToVec(cont)

    brcdIdx = brcdLst.index(brcd)

    y0 = np.zeros((1, len(brcdLst)), dtype=float)
    y0[0, brcdIdx] = 1.0
    
    # infer
    y = model(x)
    y_arg_max = torch.argmax(y)

    y0 = torch.Tensor(y0)
    loss = loss_fn(y, y0)

    # stat
    totCnt += 1
    if brcdIdx == y_arg_max:  
        crctCnt += 1
    crctRat = crctCnt / totCnt
    if cnt % 10000 == 0:
        print()
        print(f'cnt = {cnt}')
        print(f'crctRat = {crctRat}')
        print(f'loss = {loss}')
        print(f'y = {y}')
        print(f'y0 = {y0}')
        print(f'cont = {cont}')
        totCnt = 0
        crctCnt = 0
        torch.save(model.state_dict(), 'voc-train.pt')

    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()











