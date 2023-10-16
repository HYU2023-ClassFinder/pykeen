from collections import Counter

f = open("tagClaims_v4.txt", "r")

entities = []

for line in f:
    entities.append(line.split()[0])
    entities.append(line.split()[2])

entities = Counter(entities)
lowCountEntities = []

# f2 = open("tagClaims_v4_count.txt", "w")
# for key, value in entities.items():
#     f2.write(key + '\t' + str(value) + '\n')

num = "zscoreUpper40"
for key, value in entities.items():
    if(value < 16):
        lowCountEntities.append(key)

entities = list(entities)

f3 = open("tagClaims_v" + str(num) + ".txt", "w")
# print(lowCountEntities)


f4 = open("tagClaims_v4.txt", "r")

for line in f4:
    # print(line)
    if(line.split()[0] in lowCountEntities or line.split()[2] in lowCountEntities):
        continue
    else:
        f3.write(line)

f.close()
f3.close()
f4.close()