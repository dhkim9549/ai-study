import matplotlib.pyplot as plt
import numpy as np

fileLst = ['review-bert-attention-out.txt', 'review-bert-attention-out-2.txt']

yLst = []
xWin = []

for reviewFile in fileLst:
    f = open(reviewFile, "r")
    str = ''
    for line in f:
        if 'crct' not in line:
            continue 
        tokenLst = line.split(' ')
        x = float(tokenLst[2])
        if x > 0.99 or x < 0.6:
            continue
        xWin.append(x)
        if len(xWin) > 100: 
            xWin.pop(0)
        yLst.append(np.average(xWin))

xLst = np.arange(0, len(yLst), 1)
plt.plot(xLst, yLst)
plt.savefig('/var/www/html/images/plot.png')
print(yLst[-10:])
