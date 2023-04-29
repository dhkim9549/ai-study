##################################################
# IMDB review pos/neg prediction using attention model evaluation
# input v.size() = (L, E)
#    L : Source sequece length
#    E : Embedding dimension of the source (embed_size)
##################################################

from os import listdir
from os.path import isfile, join
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

    aM = torch.zeros((L, L), dtype=torch.bool) # attn_mask
    if j < L:
        aM[j:, j:] = True

    return x, aM 

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

    def forward(self, x, mL):
        xe = self.tok_embed(x)
        pe = self.pos_enc(xe)
        z = self.norm(pe) 

        q = self.Wq(z)
        k = self.Wk(z)
        v = self.Wv(z)
        y, y_w = self.self_attention_context1(q, k, v, attn_mask=mL)

        ya = torch.flatten(y)
        y2 = self.linear_relu_stack(ya)

        return y2

model = NeuralNetwork()
print(f'model = {model}')

model.load_state_dict(torch.load('train-review-torch.pt'))
model.eval()

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

loss_fn = nn.MSELoss()

# evaluate 
fileLst = []

pathPos = '/root/data/aclImdb/test/pos'
filesPos = [f for f in listdir(pathPos) if isfile(join(pathPos, f))]
for fileNm in filesPos:
    fileLst.append(pathPos + '/' + fileNm)

pathNeg = '/root/data/aclImdb/test/neg'
filesNeg = [f for f in listdir(pathNeg) if isfile(join(pathNeg, f))]
for fileNm in filesNeg:
    fileLst.append(pathNeg + '/' + fileNm)

crctCnt = 0
totCnt = 0
for cnt in range(100000000000000000000):

    reviewFile = fileLst[np.random.randint(len(fileLst))]
    f = open(reviewFile, "r")
    str = ''
    for line in f:
        str += line

    x, aM = strToVec(str)

    y0 = np.array([1.0, 0.0])
    if 'neg' in reviewFile:
        y0 = np.array([0.0, 1.0])

    # infer
    y = model(x, aM)

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
        print(f'crctRat = {crctRat}')
        print(f'loss = {loss}')
        print(f'y = {y}')
        print(f'y0 = {y0}')
        print(f'reviewFile = {reviewFile}')
        print(model.Wv.state_dict())
        totCnt = 0
        crctCnt = 0



