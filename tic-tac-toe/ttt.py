import numpy as np
import torch
from torch import nn
from torch.distributions.dirichlet import Dirichlet
import logging
import datetime

nnName = 'train-tic'

logging.basicConfig(filename='logs/' + nnName + '.log',
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.DEBUG)

g_i = 0

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

def isOver(board):
    s = np.sum(np.absolute(board.flatten()))
    if s >= 9:
        return True
    if hasWon(board) or hasWon(- board):
        return True
    return False

def getModelValue(pBoard):
    x = getX(pBoard)
    y = model(x)
    y = torch.softmax(y, dim = 0)[0]
    return y

def getTSValue(pBoard, depth=0):
    
    if hasWon(pBoard):
        return 1
    if hasWon(- pBoard):
        return 0
    if isOver(pBoard):
        return 0.5

    if depth >= 0:
        y = getModelValue(pBoard) 
        return y

    maxY = -999999999
    maxI = -1
    for i in range(9):
        board = np.array(pBoard)
        board *= -1
        if board[divmod(i, 3)] != 0:
            continue
        board[divmod(i, 3)] = 1;
        y = getTSValue(board, depth = depth + 1)
        if y > maxY:
            maxY = y

    return 1 - maxY

"""
bb = np.zeros((3, 3))
bb[1, 1] = -1
bb[0, 0] = 1
vv = getValue(bb)
print(f'vv = {vv}')

bb = np.zeros((3, 3))
bb[0, 1] = -1
bb[1, 1] = 1
vv = getValue(bb)
print(f'vv = {vv}')
"""

def getAction(pBoard, display=False, epsilon=0.0, isTreeSearch=False):

    matrix = np.zeros((3, 3))
    maxY = -999999999
    maxI = -1
    for i in range(9):
        board = np.array(pBoard)
        if board[divmod(i, 3)] != 0:
            continue
        board[divmod(i, 3)] = 1;
        if isTreeSearch == True:
            y = getTSValue(board)
        else:
            y = getModelValue(board)

        matrix[divmod(i, 3)] = y

    m = Dirichlet(torch.ones(3, 3) * 0.03)
    noise = m.sample().detach().numpy()

    yy = (1 - epsilon) * matrix + epsilon * noise

    for i in range(9):
        y = yy[divmod(i, 3)]
        if y > maxY:
            maxY = y
            maxI = i

    if display == True:
        logging.info(f'matrix =\n{matrix}')

    return maxI

def getRandomAction(board):
    while True:
        a = np.random.randint(0, 9)
        if board[divmod(a, 3)] == 0:
            return a
    return -1

def getAction2(board):
    return getAction(board, epsilon=0.25, isTreeSearch=False)

def getAction3(board):
    return getAction(board, epsilon=0.0, isTreeSearch=False)

def play(action1, action2):
    board = np.zeros((3, 3), dtype=np.int16)
    boardArr = []
    score = 0
    i = 0
    while not isOver(board):
        a = action1(board) if int(i) % 2 == 0 else action2(board)
        board[divmod(a, 3)] = 1
        boardArr.append(np.copy(board))
        board *= -1
        i += 1
    if int(i) % 2 == 1:
        board *= -1
    if hasWon(board):
        score = 1
    elif hasWon(- board):
        score = -1

    return (score, boardArr)

def evaluate():
    win, tie, lose, cnt = 0, 0, 0, 0
    for i in range(10000):
        score = 0
        cnt += 1
        if int(i) % 2 == 0:
            score, boardArr = play(getAction3, getRandomAction)
            if score > 0:
                win += 1
            elif score < 0:
                lose += 1
            else:
                tie += 1
        else:
            score, boardArr = play(getRandomAction, getAction3)
            if score > 0:
                lose += 1
            elif score < 0:
                win += 1
            else:
                tie += 1
    logging.info(f'evaluation = {(win, tie, lose, cnt)}') 

    board = np.zeros((3, 3), dtype=np.int16)
    getAction(board, display=True)

    board = np.zeros((3, 3), dtype=np.int16)
    board[0, 1] = -1
    getAction(board, display=True)

    board = np.zeros((3, 3), dtype=np.int16)
    board[1, 1] = -1
    board[0, 0] = 1
    board[0, 1] = -1
    getAction(board, display=True)

for i in range(100000000000000):

    g_i = i

    if i % 1000 == 0:
        logging.info(f'i = {i}')
    if i % 30000 == 0:
        evaluate()
        PATH = f'./models/model-TS-{i}.pth'
        torch.save(model.state_dict(), PATH)

    score, boardArr = play(getAction2, getAction2)

    # sample train data
    r = np.random.randint(0, len(boardArr))

    x = getX(boardArr[r])
    y0 = score * (1 if int(r) % 2 == 0 else -1)
    if (y0 - 1) ** 2 < 0.001:
        y0 = torch.tensor([1, 0], dtype=torch.float32)
    elif (y0 + 1) ** 2 < 0.001:
        y0 = torch.tensor([0, 1], dtype=torch.float32)
    else:
        y0 = torch.tensor([0.5, 0.5], dtype=torch.float32) 

    y = model(x)
    loss = loss_fn(y, y0)
    if i % 100001 == 0:
        logging.info((x, y, y0))
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    



