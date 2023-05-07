"""
IMDB review pos/neg prediction using DistilBERT with pooled output and linear model
   L : Source sequece length
   embed_size : Embedding dimension of the source
"""

from os import listdir
from os.path import isfile, join
import datetime
import numpy as np
import torch
from torch import nn
import PosEnc
from transformers import DistilBertTokenizer, DistilBertModel

embed_size = 768 
embed_dim = 70 
L = 100 

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distbert = DistilBertModel.from_pretrained("distilbert-base-uncased")

def distbert_enc(input_text):
    aM = torch.zeros((L, L), dtype=torch.bool) # attn_mask
    encoded_input = tokenizer(input_text, return_tensors='pt')

    x = encoded_input.input_ids
    m = x.size()[1] 
    if m > L:
        x = x[:, :L]
    if m < L:
        aM[m:, m:] = True
        xsize_diff = L - x.size()[1]
        x_pad = torch.zeros((1, xsize_diff), dtype=int)
        x = torch.cat((x, x_pad), dim=1)
    output = distbert(input_ids=x)
    y = output.last_hidden_state
    y = torch.squeeze(y)
    return y, aM 

str_dict = {}

# truncate string if the list of tokens is too long
def truncateStr(input_str):
    x = input_str.split(' ')
    y = ''
    for i in range(len(x)):
        if i > 350:
            break
        y += x[i] + ' '
    return y 

# Converts str to tensor
def strToVec(input_str):
    if input_str in str_dict:
        return str_dict[input_str]
    else:
        x, aM = distbert_enc(input_str)
        x = torch.flatten(x)
        str_dict[input_str] = x.detach()
    return x

# nn
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        self.linear_relu_stack = nn.Sequential(
            nn.LazyLinear(100),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.LazyLinear(2),
            nn.Softmax()
        )

    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y

model = NeuralNetwork()
print(f'model = {model}')

#model.load_state_dict(torch.load('review-bert-linear.pt'))

learningRat = 0.002
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRat)

# train
fileLst = []

pathPos = '/data/aclImdb/train/pos'
filesPos = [f for f in listdir(pathPos) if isfile(join(pathPos, f))]
for fileNm in filesPos:
    fileLst.append(pathPos + '/' + fileNm)

pathNeg = '/data/aclImdb/train/neg'
filesNeg = [f for f in listdir(pathNeg) if isfile(join(pathNeg, f))]
for fileNm in filesNeg:
    fileLst.append(pathNeg + '/' + fileNm)

crctCnt = 0
totCnt = 0
for cnt in range(100000000000000000000):
    reviewFile = fileLst[np.random.randint(len(fileLst))]

    f = open(reviewFile, "r")
    s = ''
    for line in f:
        s += line.replace('<br />', ' ') + ' '

    s = truncateStr(s)
    x = strToVec(s)

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
    if cnt % 100 == 0:
        print()
        print(f'cnt = {cnt}')
        print(datetime.datetime.now())
        print(f'crctRat = {crctRat}')
        print(f'loss = {loss}')
        print(f'y = {y}')
        print(f'y0 = {y0}')
        print(f'reviewFile = {reviewFile}')
        print(f'len(str_dict) = {len(str_dict)}')
        totCnt = 0
        crctCnt = 0
        if cnt % 1000 == 0:
            torch.save(model.state_dict(), 'review-bert-linear.pt')

    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


