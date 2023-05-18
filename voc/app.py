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

brcdLst = ['고객만족부', '지사', '정책모기지부', '주택보증부', '유동화자산부', '종합금융센터', '주택연금부', 'ICT운영부', '인사부', '유동화증권부', '홍보실', '주택금융연구원', '사업자보증부', '경영혁신부', '채권관리부']

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

model.load_state_dict(torch.load('voc-train.pt'))
model.eval()

loss_fn = nn.CrossEntropyLoss()

logging.basicConfig(filename = "logs/project.log", level = logging.DEBUG)
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

@app.route('/predict')
def predict():
    cont = request.args.get('cont')

    x = strToVec(cont)

    # infer
    y = model(x)

    rsltLst = [] 
    yLst = y[0].tolist()
    i = 0
    for t in yLst:
        rsltDict = {}
        rsltDict['deptNm'] = brcdLst[i]
        rsltDict['scor'] = t
        rsltLst.append(rsltDict)
        i += 1

    logging.info(rsltLst)

    return jsonify(rsltLst)





