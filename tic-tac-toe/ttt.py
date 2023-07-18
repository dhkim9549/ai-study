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
            nn.LazyLinear(100),
            nn.ReLU(),
            nn.LazyLinear(100),
            nn.ReLU(),
            nn.LazyLinear(1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0002)

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
    board = np.array(board)
    maxY = -999999999
    maxI = -1
    for i in range(9):
        if board[divmod(i, 3)] != 0:
            continue
        board[divmod(i, 3)] = 1;
        x = getX(board)
        y = model(x)
        if y > maxY:
            maxY = y
            maxI = i
    return maxI

def getMatrix():
    matrix = np.zeros((3, 3))
    for i in range(9):
        board = np.zeros((3, 3), dtype=np.int16)
        board[divmod(i, 3)] = 1;
        matrix[divmod(i, 3)] = model(getX(board)) 
    print(f'matrix = {matrix}')

def getRandomAction(board):
    while True:
        a = np.random.randint(0, 9)
        if board[divmod(a, 3)] == 0:
            return a
    return -1

def getAction2(board):
    r = np.random.random()
    if r <= 0.3:
        return getRandomAction(board)
    else:
        return getAction(board)

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
    for i in range(1000):
        score = 0
        cnt += 1
        if int(i) % 2 == 0:
            score, boardArr = play(getAction, getRandomAction)
            if score > 0:
                win += 1
            elif score < 0:
                lose += 1
            else:
                tie += 1
        else:
            score, boardArr = play(getRandomAction, getAction)
            if score > 0:
                lose += 1
            elif score < 0:
                win += 1
            else:
                tie += 1
    print (win, tie, lose, cnt)

    getMatrix()




X = None
Y0 = None

for i in range(100000000000000):

    score, boardArr = play(getAction2, getAction2)

    # sample train data
    r = np.random.randint(0, len(boardArr))
    x = getX(boardArr[r])
    y0 = score * (1 if int(r) % 2 == 0 else -1)
    y0 = torch.tensor(y0, dtype=torch.float32)
    if int(i) % 16 == 0:
        X = torch.tensor(x)
        Y0 = torch.tensor(y0)
    else:
        X = torch.vstack((X, x))
        Y0 = torch.vstack((Y0, y0))

    if int(i) % 16 == 15:
        Y = model(X)
        loss = loss_fn(Y, Y0)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if i % 10000 == 0:
        print(f'i = {i}')
        evaluate()



