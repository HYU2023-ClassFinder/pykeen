import csv
 
f = open('output.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)

prereqList = []

for line in rdr:
    a = line[0]
    b = "precedes"
    c = line[1]

    prereqList.append(a + ' ' + b + ' ' + c)

print(prereqList[0])

f2 = open("tagClaims_v2_3.txt", "w")
for prereq in prereqList:
    f2.write(prereq)
    f2.write('\n')

f.close()
f2.close()