import numpy as np
import torch
from torch import nn

board = np.zeros((3, 3))
board[1, 1] = 1
board[0, 0] = -1

print(f'board = {board}')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.LazyLinear(50),
            nn.ReLU(),
            nn.LazyLinear(9),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
print(model)

def getX(board):
    x = np.maximum(board.flatten(), 0)
    y = np.maximum(- board.flatten(), 0)
    z = np.ones(9) - x - y
    c = np.concatenate((x, y, z))
    r = torch.tensor(c, dtype=torch.float32)
    return r 

x = getX(board)
print(f'x = {x}')

y = model(x)
print(f'y = {y}')

def hasWon(board):
    for i in range(3):
        p = 0
        q = 0
        for j in range(3):
            if board[i, j] == 1:
                p += 1
            if board[j, i] == 1:
                q += 1
        if p >= 3 or q >= 3:
            return True;
    d = 0
    e = 0
    for i in range(3):
        if board[i, i] == 1:
            d += 1
        if board[i, 2 - i] == 1:
            e += 1
    if d >= 3 or e >= 3:
        return True;
    return False;

w = hasWon(board);
print(f'w = {w}')

def isOver(board):
    s = np.sum(np.absolute(board.flatten()))
    if s >= 9:
        return True
    if hasWon(board) or hasWon(- board):
        return True
    return False

o = isOver(board)
print(f'o = {o}')

def getPoint(board):
    if hasWon(board):
        return 1
    if hasWon(- board):
        return -1
    return 0

def getAction(board):
    x = getX(board)
    y = model(x)
    a = torch.argmax(y)
    return a

def play():
    board = np.zeros((3, 3))

