#######################################################################
# Make a sorted list of IMDB vocabulary
#######################################################################

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

vocaLst = []
f = open("/root/data/aclImdb/imdbEr.txt", "r")
i = 0
er = {}
for x in f:
    i += 1
    token = x.split("\n")[0]
    er[i] = token
    t = (voca2[i], float(token))
    vocaLst.append(t)
    if i > 1000:
        break

print(f'vocaLst = {vocaLst}')

def compare(item1, item2):
    return item1[1] - item2[1]

sortedLst = sorted(vocaLst, key=lambda x: x[1])
print(f'sortedLst = {sortedLst}')

for i in range(100000):
    print(sortedLst[i])


