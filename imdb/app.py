###################################################
# Serve nn as an REST API with Flask
###################################################

from os import listdir
from os.path import isfile, join
import numpy as np
import torch
from torch import nn

from flask import Flask, jsonify, request
from flask_cors import CORS

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
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(len(voca), 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Softmax()
        )

    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y

model = NeuralNetwork()
print(f'model = {model}')

model.load_state_dict(torch.load('train-review-torch.pt'))
model.eval()

loss_fn = nn.MSELoss()

app = Flask(__name__)
CORS(app)

@app.route('/predict')
def predict():
    cont = request.args.get('cont')
    tokenLst = cont.replace(".", "").replace("!", "").lower().split(" ") 
    x = np.zeros((1, len(voca)), dtype=float)

    for token in tokenLst:
        if token in voca:
            i = voca[token]
            x[0, i] += 1
    x = np.minimum(x, 1.0)
    x = torch.Tensor(x)

    # infer
    y = model(x)

    return jsonify({'output': y[0].tolist()})





