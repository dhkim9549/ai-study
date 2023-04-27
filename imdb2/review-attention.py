##################################################
# IMDB review pos/neg prediction using attention model
# input v.size() = (L, E)
#    L : Source sequece length
#    E : Embedding dimension of the source (embed_size)
##################################################

from os import listdir
from os.path import isfile, join
import datetime
import numpy as np
import torch
from torch import nn
import PosEnc

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
    if(i >= 1000):
        break

embed_size = 64 
embed_dim = embed_size 
L = 100 

# Converts str to tensor
def strToVec(str):
    tokenLst = []
    tokenLst += str.replace(".", "").replace("!", "").lower().split(" ")

    x = np.zeros(L, dtype=int)

    j = 0
    for token in tokenLst:
        if j >= L:
            break
        if token in voca:
            i = voca[token]
            x[j] = i
        j += 1

    x = torch.tensor(x, dtype=torch.int32)

    return x 

# nn
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.tok_embed = nn.Embedding(len(voca), embed_size)  # token embedding
        self.pos_enc = PosEnc.PositionalEncoding(embed_size, 0, L)  # position encoding 
        self.norm = nn.LayerNorm(embed_size)

        self.Wq = nn.LazyLinear(embed_size)
        self.Wk = nn.LazyLinear(embed_size)
        self.Wv = nn.LazyLinear(embed_dim)
        self.self_attention_context1 = nn.MultiheadAttention(embed_dim, 1, dropout=0.5)
        
        self.linear_relu_stack = nn.Sequential(
            nn.LazyLinear(50),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.LazyLinear(2),
            nn.Softmax()
        )

    def forward(self, x):
        xe = self.tok_embed(x)
        pe = self.pos_enc(xe)
        z = self.norm(pe) 

        q = self.Wq(z)
        k = self.Wk(z)
        v = self.Wv(z)
        y, y_w = self.self_attention_context1(q, k, v)

        y2 = self.linear_relu_stack(torch.flatten(y))

        return y2

model = NeuralNetwork()
print(f'model = {model}')

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

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
flag = True 
for cnt in range(100000000000000000000):
    reviewFile = fileLst[np.random.randint(len(fileLst))]

    f = open(reviewFile, "r")
    str = ''
    for line in f:
        str += line

    x = strToVec(str)

    y0 = np.array([1.0, 0.0])
    if 'neg' in reviewFile:
        y0 = np.array([0.0, 1.0])

    # infer
    y = model(x)

    y0 = torch.Tensor(y0)
    loss = loss_fn(y, y0)

    # stat
    totCnt += 1
    if (y[0] > 0.5 and y0[0] > 0.5) or (y[0] <  0.5 and y0[0] <  0.5):
        crctCnt += 1
    crctRat = crctCnt / totCnt 
    if cnt % 1000 == 0:
        print()
        print(f'cnt = {cnt}')
        print(datetime.datetime.now())
        print(f'crctRat = {crctRat}')
        print(f'loss = {loss}')
        print(f'y = {y}')
        print(f'y0 = {y0}')
        print(f'reviewFile = {reviewFile}')
        print(model.Wv.state_dict())
        totCnt = 0
        crctCnt = 0
        torch.save(model.state_dict(), 'train-review-torch.pt')
        if crctRat > 0.7 and flag:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.002 
            flag = False 

    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


