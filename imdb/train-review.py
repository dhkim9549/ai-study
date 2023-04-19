from os import listdir
from os.path import isfile, join
import numpy as np

# Load vocab
f = open("/root/data/aclImdb/imdb.vocab", "r")
i = 0
voca = {} 
voca2 = {}
for x in f:
    i += 1
    token = x.split("\n")[0]
    voca[token] = i
    voca2[i] = token
    if i > 1000:
        break

# train
mypath = '/root/data/aclImdb/train/pos'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

if(1 == 1):
    reviewFile = onlyfiles[0]

    f = open(mypath + '/' + reviewFile, "r")
    tokenLst = []
    for line in f:
        tokenLst += line.replace(".", "").replace("!", "").lower().split(" ")
    print(tokenLst)

    x = np.zeros((1, 1000), dtype=float)
    print(f'x = {x}')

    for token in tokenLst:
        if token in voca:
            i = voca[token]
            x[0, i] += 1

    print(reviewFile)
    print(f'x = {x}')


