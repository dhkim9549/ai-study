##################################################
# Preprocess VOC data
#     make a vocab
##################################################

brcdMap = {}
dataDict = {}
vocabDict = {}

# Make vocab
f = open("/data/v-xy19-21/20230406_VOC_XY_DATA_200000-210000.txt", "r")
i = 0
for x in f:
    i += 1
    if i == 1:
        continue

    x = x.replace('\n', '').replace('.', '').replace(',', '')

    tokens = x.split("\t")
    brcd = tokens[8]
    cont = tokens[3] + ' ' + tokens[4]

    words = cont.split(' ')
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

vocabDict = {k: v for k, v in sorted(vocabDict.items(), key=lambda item: - item[1])}

f = open("voc-vocab.txt", "w")
i = 0
for w in vocabDict:
    f.write(w + '\n')

    i += 1
    if i > 1000000000:
        break
f.close()







