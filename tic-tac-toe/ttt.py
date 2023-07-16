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

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.002)

def getX(board):
    x = np.maximum(board.flatten(), 0)
    y = np.maximum(- board.flatten(), 0)
    z = np.ones(9) - x - y
    c = np.concatenate((x, y, z))
    r = torch.tensor(c, dtype=torch.float32)
    return r 

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

def isOver(board):
    s = np.sum(np.absolute(board.flatten()))
    if s >= 9:
        return True
    if hasWon(board) or hasWon(- board):
        return True
    return False

o = isOver(board)

def getPoint(board):
    if hasWon(board):
        return 1
    if hasWon(- board):
        return -1
    return 0

def getAction(board):
    x = getX(board)
    y = model(x)
    ti = torch.topk(y, 9).indices
    for i in range(9):
        a = int(ti[i])
        if board[divmod(a, 3)] == 0:
            return a
    return -1 

def getRandomAction(board):
    while true:
        a = np.random.randint(0, 9)
        if board[divmod(a, 3)] == 0:
            return a
    return -1

def evaluate(model):
    for i in range(1):
        board = np.zeros((3, 3), dtype=np.int16)
        s = int(i) % 2 == 0 ? 1 : -1
        while True:
            if s == 1:
                a = getAction(board)
            board[divmod(a, 3)] = 1





def play():
    board = np.zeros((3, 3), dtype=np.int16)
    boardArr = []
    actionArr = []
    i = 0
    while True:
        a = getAction(board)
        boardArr.append(np.copy(board))
        actionArr.append(a)

        board[divmod(a, 3)] = 1

        if isOver(board):
            break

        board *= -1
        i += 1

    if not hasWon(board):
        return

    # sample train data
    r = np.random.randint(0, len(boardArr))

    x = getX(boardArr[r])
    y = model(x)
    y0 = torch.zeros(9)
    y0[actionArr[r]] = 1
    if (int(i) % 2 == 0 and int(r) % 2 == 1) or (int(i) % 2 == 1 and int(r) % 2 == 0):
        y0 = 1 - y0
    print(f'y0 = {y0}')

    loss = loss_fn(y, y0)
    print(f'loss = {loss}')

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

for i in range(100):
    play()
