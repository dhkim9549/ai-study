"""
IMDB review pos/neg prediction using DistilBERT and attention model
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
    encoded_input = tokenizer(input_text, return_tensors='pt')
    x = encoded_input.input_ids
    if x.size()[1] > L:
        x = x[:, :L]
    if x.size()[1] < L:
        xsize_diff = L - x.size()[1]
        x_pad = torch.zeros((1, xsize_diff), dtype=int) + 102
        x = torch.cat((x, x_pad), dim=1)
    output = distbert(input_ids=x)
    y = output.last_hidden_state
    y = torch.squeeze(y)
    return y 

# Converts str to tensor
def strToVec(input_str):
    x = distbert_enc(input_str)
    aM = torch.zeros((L, L), dtype=torch.bool) # attn_mask
    """
    if j < L:
        aM[j:, j:] = True
    """

    return x, aM 

# nn
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.pos_enc = PosEnc.PositionalEncoding(embed_size, 0, L)  # position encoding 
        self.norm = nn.LayerNorm(embed_size)

        self.Wq = nn.LazyLinear(embed_dim)
        self.Wk = nn.LazyLinear(embed_dim)
        self.Wv = nn.LazyLinear(embed_dim)
        self.self_attention_context1 = nn.MultiheadAttention(embed_dim, 1, dropout=0.5)
        
        self.linear_relu_stack = nn.Sequential(
            nn.LazyLinear(100),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.LazyLinear(2),
            nn.Softmax()
        )

    def forward(self, x, aM):
        pe = self.pos_enc(x)
        z = self.norm(pe) 

        q = self.Wq(z)
        k = self.Wk(z)
        v = self.Wv(z)
        y, y_w = self.self_attention_context1(q, k, v, attn_mask=aM)

        ya = torch.flatten(y)
        y2 = self.linear_relu_stack(ya)

        return y2

model = NeuralNetwork()
print(f'model = {model}')

learningRat = 0.02
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRat)

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
    if cnt % 100 == 0:
        print()
        print(f'cnt = {cnt}')
        print(datetime.datetime.now())
        print(f'crctRat = {crctRat}')
        print(f'loss = {loss}')
        print(f'y = {y}')
        print(f'y0 = {y0}')
        print(f'reviewFile = {reviewFile}')
        totCnt = 0
        crctCnt = 0
        if cnt % 10000 == 0:
            torch.save(model.state_dict(), 'review-bert-attention.pt')
        if crctRat > 0.7 and learningRat > 0.002:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.002 
            flag = False 

    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


