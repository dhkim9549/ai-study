import numpy as np
import torch
from torch import nn
import logging
import datetime

nnName = 'serve-tic'

logging.basicConfig(filename='logs/' + nnName + '.log',
                    filemode='w',
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

file_path = './models/model-TS-2905000.pth'
model.load_state_dict(torch.load(file_path))
model.eval()

sBoard = torch.zeros(3, 3)
for i in range(3):
    for j in range(3):
        board = np.zeros((3, 3), dtype=np.int16)
        board[0][0] = -1
        board[0][1] = -1
        board[1][1] = 1
        board[1][0] = 1
        if board[i][j] != 0:
            continue
        board[i][j] = 1
        x = getX(board)
        print(f'x = {x}')
        y = model(x)
        print(f'y = {y}')
        y = torch.softmax(y, dim = 0)[0]
        print(f'y = {y}')
        sBoard[i][j] = y

print(f'sBoard = {sBoard}')



