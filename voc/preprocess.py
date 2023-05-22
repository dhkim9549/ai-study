##################################################
# Preprocess VOC data
#     make a vocab
##################################################

import re

brcdMap = {}
dataDict = {}
vocabDict = {}

# Make vocab
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

    if vocDy < '20200101':
        continue

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

    cont = cont.replace('&#039;', ' ').replace('\n', ' ').replace('&quot;', ' ')
    cont = re.sub(r'[:\*\?/\(\)\[\]~\.,\\？!]', ' ', cont)

    words = cont.split()
    for w in words:
        if w == '':
            continue
        if w in vocabDict:
            vocabDict[w] += 1
        else:
            vocabDict[w] = 1

    if brcd in brcdMap:
        brcdMap[brcd] += 1
    else:
        brcdMap[brcd] = 1
    if brcd in dataDict:
        dataLst = dataDict[brcd]
        dataLst.append(cont)
    else:
        dataLst = []
        dataLst.append(cont)
        dataDict[brcd] = dataLst

    if i > 4000000000:
        break

x = brcdMap
x = {k: v for k, v in sorted(x.items(), key=lambda item: - item[1])}
print(x)

vocabDict = {k: v for k, v in sorted(vocabDict.items(), key=lambda item: - item[1])}
vocabSet = set() 

f = open("voc-vocab.txt", "w")
i = 0
for w in vocabDict:
    f.write(w + '\t' + str(vocabDict[w]) + '\n')
    vocabSet.add(w)

    i += 1
    if i >= 40000:
        break
f.close()

print(f'len(vocabSet) = {len(vocabSet)}')
