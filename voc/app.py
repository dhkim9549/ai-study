###################################################
# Serve nn as an REST API with Flask
###################################################

import re
from os import listdir
from os.path import isfile, join
import numpy as np
import torch
from torch import nn

from flask import Flask, jsonify, request
from flask_cors import CORS
import logging

def softmax(x):
    return(np.exp(x)/np.exp(x).sum())

# Load vocab
f = open("voc-vocab.txt", "r")
i = 0
voca = {}
voca2 = {}
for x in f:
    token = x.split()[0]
    voca[token] = i
    voca2[i] = token
    i += 1
    if(i >= 40000):
        break

# Cenverts str to numpy array
def strToVec(s):
    s = s.replace('&#039;', ' ').replace('&quot;', ' ')
    s = re.sub(r'[:\*\?/\(\)\[\]~\.,\\？!\n\t]', ' ', s)

    x = np.zeros((1, len(voca)), dtype=float)

    tokenLst = s.split(" ")

    for token in tokenLst:
        if token in voca:
            i = voca[token]
            x[0, i] += 1
    x = np.minimum(x, 1.0)
    x = torch.Tensor(x)

    return x

brcdLst = ['정책모기지부', '지사', '유동화자산부', '주택보증부', '주택연금부', 'ICT운영부', '종합금융센터', '유동화증권부', '사업자보증부', '채권관리부', '인사부', '경영혁신부', '홍보실', '주택금융연구원']

# nn
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.LazyLinear(200),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LazyLinear(len(brcdLst)),
        )

    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y

model = NeuralNetwork()
print(f'model = {model}')

model.load_state_dict(torch.load('pt/train-voc-nc-v40000-h200.pt'))
model.eval()

loss_fn = nn.CrossEntropyLoss()

logging.basicConfig(filename = "logs/app.log", level = logging.DEBUG)
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

@app.route('/predict')
def predict():
    cont = request.args.get('cont')

    x = strToVec(cont)
    logging.info(f'x = {x}')

    # infer
    y = model(x)
    logging.info(f'y = {y}')

    rsltLst = [] 
    yLst = y[0].tolist()
    yLst = softmax(yLst)
    logging.info(f'yLst = {yLst}')

    i = 0
    for t in yLst:
        rsltDict = {}
        rsltDict['deptNm'] = brcdLst[i]
        rsltDict['scor'] = t
        rsltLst.append(rsltDict)
        i += 1

    logging.info(f'rsltLst = {rsltLst}')

    return jsonify(rsltLst)





