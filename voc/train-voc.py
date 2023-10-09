##################################################
# Train VOC data
##################################################

import re
import numpy as np
import torch
from torch import nn
import logging
import datetime

nnName = 'train-voc-nc-v40000-h200'

logging.basicConfig(filename='logs/' + nnName + '.log',
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

vocDict = {}
brcdMap = {}
trainDict = {}
testDict = {}

brcdLst = ['정책모기지부', '지사', '유동화자산부', '주택보증부', '주택연금부', 'ICT운영부', '종합금융센터', '유동화증권부', '사업자보증부', '채권관리부', '인사부', '경영혁신부', '홍보실', '주택금융연구원']

# Load training data 
f = open("/data/voc-all/20230518_VOC_XY_DATA_ALL.txt", "r")
i = 0
for x in f:
    i += 1
    if i == 1:
        continue

    tokens = x.split("\t")
    if len(tokens) != 9:
        continue

    vocDy = tokens[1]
    brcd = tokens[8]
    cont = tokens[3] + ' ' + tokens[4]

    brcd = brcd.strip()
    if '팀' in brcd:
        continue
    if brcd == '디지털금융부':
        continue
    if brcd == '정책모기지서비스센터':
        continue
    if brcd == '사회적가치부':
        brcd = '고객만족부'
    if brcd == '채권관리센터':
        brcd = '종합금융센터'
    if '지사' in brcd:
        brcd = '지사'

    if vocDy < '20200101':
        continue
    if brcd not in brcdLst:
        continue

    cont = tokens[3] + ' ' + tokens[4]

    if vocDy >= '20230301':
        testDict[cont] = brcd
        continue

    if brcd in brcdMap:
        brcdMap[brcd] += 1
    else:
        brcdMap[brcd] = 1
    if brcd in vocDict:
        dataLst = vocDict[brcd]
        dataLst.append(cont)
    else:
        dataLst = []
        dataLst.append(cont)
        vocDict[brcd] = dataLst

tempMap = {}
for k in vocDict:
    tempMap[k] = len(vocDict[k]) 
logging.info(f'tempMap = {tempMap}')

logging.info(f'brcdLst = {brcdLst}')
logging.info(f'len(brcdLst) = {len(brcdLst)}')

for brcd in vocDict:
    for cont in vocDict[brcd]:
        trainDict[cont] = brcd

logging.info(f'len(trainDict) = {len(trainDict)}')

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

logging.info(f'len(voca) = {len(voca)}')

# Cenverts str to numpy array
def strToVec(s):
    s = s.replace('&#039;', ' ').replace('&quot;', ' ')
    s = re.sub(r'[:\*\?/\(\)\[\]~\.,\\？!‘’]', ' ', s)

    x = np.zeros((1, len(voca)), dtype=float)

    tokenLst = s.split()

    for token in tokenLst:
        if token in voca:
            i = voca[token]
            x[0, i] += 1
    x = np.minimum(x, 1.0)
    x = torch.Tensor(x)

    return x

def evaluate():

    logging.info('Evaluating...')

    totCntEval = 0
    crctCntEval = 0

    y0_cnt_eval = np.zeros((1, len(brcdLst)), dtype=int)
    y_cnt_eval = np.zeros((1, len(brcdLst)), dtype=int)

    for cont in testDict: 

        brcd = testDict[cont] 

        x = strToVec(cont)

        brcdIdx = brcdLst.index(brcd)

        y0 = np.zeros((1, len(brcdLst)), dtype=float)
        y0[0, brcdIdx] = 1.0
        y0_cnt_eval[0, brcdIdx] += 1
        
        # infer
        model.eval()
        y = model(x)
        model.train()

        y_arg_max = torch.argmax(y)
        y_cnt_eval[0, y_arg_max] += 1
        brcd_infer = brcdLst[y_arg_max]

        y0 = torch.Tensor(y0)

        # stat
        totCntEval += 1
        if (brcdIdx - y_arg_max) ** 2 < 0.01:  
            crctCntEval += 1
        crctRatEval = crctCntEval / totCntEval

    logging.info(f'totCntEval = {totCntEval}')
    logging.info(f'crctRatEval = {crctRatEval}')
    logging.info(f'y = {y}')
    logging.info(f'y0 = {y0}')
    logging.info(f'y0_cnt_eval = {y0_cnt_eval}')
    logging.info(f'y_cnt_eval = {y_cnt_eval}')
    logging.info('Evaluation end...')

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
# model.load_state_dict(torch.load('voc-train.pt'))

logging.info(f'model = {model}')
loss_fn = nn.CrossEntropyLoss()
logging.info(f'loss_fn = {loss_fn}')
optimizer = torch.optim.SGD(model.parameters(), lr=0.002)
logging.info(f'optimizer = {optimizer}')

crctCnt = 0
totCnt = 0

contLst = list(trainDict.keys())

y0_cnt = np.zeros((1, len(brcdLst)), dtype=int)
y_cnt = np.zeros((1, len(brcdLst)), dtype=int)

try:
    for cnt in range(1000001):

        cont = contLst[np.random.randint(len(contLst))]
        brcd = trainDict[cont] 

        """
        brcd = brcdLst[np.random.randint(len(brcdLst))]
        contLst = vocDict[brcd]
        cont = contLst[np.random.randint(len(contLst))]
        """

        x = strToVec(cont)

        brcdIdx = brcdLst.index(brcd)

        y0 = np.zeros((1, len(brcdLst)), dtype=float)
        y0[0, brcdIdx] = 1.0
        y0_cnt[0, brcdIdx] += 1
        
        # infer
        y = model(x)
        y_arg_max = torch.argmax(y)
        y_cnt[0, y_arg_max] += 1
        brcd_infer = brcdLst[y_arg_max]

        y0 = torch.Tensor(y0)

        # stat
        totCnt += 1
        if (brcdIdx - y_arg_max) ** 2 < 0.01:  
            crctCnt += 1
        crctRat = crctCnt / totCnt
        if cnt % 10000 == 0:
            logging.info(datetime.datetime.now())
            logging.info(f'cnt = {cnt}')
            logging.info(f'crctRat = {crctRat}')
            logging.info(f'y = {y}')
            logging.info(f'y0 = {y0}')
            logging.info(f'cont = {cont}')
            logging.info(f'brcd = {brcd}')
            logging.info(f'brcd_infer = {brcd_infer}')
            logging.info(f'y0_cnt = {y0_cnt}')
            logging.info(f'y_cnt = {y_cnt}')

            evaluate()

            logging.info('--------------------------------------------------------------------------------')

            totCnt = 0
            crctCnt = 0
            y0_cnt = np.zeros((1, len(brcdLst)), dtype=int)
            y_cnt = np.zeros((1, len(brcdLst)), dtype=int)

        if cnt % 100000 == 0:
            torch.save(model.state_dict(), 'pt/' + nnName + '.pt')

        # backpropagation
        loss = loss_fn(y, y0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

except Exception:
    logging.exception('Exception in training loop')

