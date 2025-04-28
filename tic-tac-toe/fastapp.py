"""
uv run fastapi dev fastapp.py --host 0.0.0.0 --port 8000
uv run fastapi run fastapp.py --port 8000
"""

from typing import Union
from fastapi import FastAPI
import json
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import torch
from torch import nn
import logging
import datetime

nnName = 'fastapp-ttt'

logging.basicConfig(filename='logs/' + nnName + '.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.DEBUG)

def getX(board):
    x = np.maximum(board.flatten(), 0)
    y = np.maximum(- board.flatten(), 0)
    z = np.ones(9) - x - y
    c = np.concatenate((x, y, z))
    r = torch.tensor(c, dtype=torch.float32)
    return r

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.LazyLinear(100),
            nn.ReLU(),
            nn.LazyLinear(100),
            nn.ReLU(),
            nn.LazyLinear(2)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
logging.info(model)

file_path = './models/ttt-lr0002-ex015-ep099-6970000.pth'
model.load_state_dict(torch.load(file_path))
model.eval()

def infer(q):
    logging.info(f'infor q = {q}')
    board = np.zeros((3, 3), dtype=np.int16)
    for i in range(3):
        for j in range(3):
            c = q[i * 3 + j]
            if c == 'o':
                board[i][j] = 1
            elif c == 'x':
                board[i][j] = -1

    sBoard = torch.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            if board[i][j] != 0:
                continue
            board[i][j] = 1
            x = getX(board)
            logging.info(f'x = {x}')
            y = model(x)
            logging.info(f'y = {y}')
            y = torch.softmax(y, dim = 0)[0]
            logging.info(f'y = {y}')
            sBoard[i][j] = y
            board[i][j] = 0
    
    logging.info(f'sBoard = {sBoard}')
    
    return sBoard;



app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World3920930293"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):

    logging.info(f'q = {q}')

    y = infer(q)
    y = y.detach().numpy().tolist()
    y = json.dumps(y)
    logging.info(f'y = {y}')

    return y;


@app.get("/greeting")
def read_greeting(board: Union[str, None] = None):

    logging.info('read_greeting() start...')

    logging.info(f'board = {board}')
    board = json.loads(board)
    logging.info(f'board = {board}')

    b = board['board']
    logging.info(f'b = {b}')

    cp = board['currentPlayer']
    logging.info(f'cp = {cp}')

    q = []
    for i, v in enumerate(b):
        if v == '':
            q.append('e')
        elif v == cp:
            q.append('o')
        else:
            q.append('x')
    logging.info(f'q = {q}')

    sBoard = infer(q)

    maxY = -999999999
    maxI = -1
    for i in range(9):
        y = sBoard[divmod(i, 3)]
        if y > maxY:
            maxY = y
            maxI = i

    y = {}
    y['a'] = maxI
    y = json.dumps(y)
    logging.info(f'y = {y}')

    return y;






