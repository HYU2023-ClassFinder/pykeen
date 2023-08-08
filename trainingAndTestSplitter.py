filename = "tagClaims_soohan_v1"
f = open(filename+".txt", "r")
train = []
valid = []
test = []

cnt = 0

while True:
    line = f.readline()
    if not line: break 
    if(cnt%10 == 0):
        valid.append(line)
    elif(cnt%10==1):
        test.append(line)
    else:
        train.append(line)

    cnt = cnt+1

trainText = open("src/pykeen/datasets/CS/"+filename+"_train.txt", "w")
validText = open("src/pykeen/datasets/CS/"+filename+"_valid.txt", "w")
testText = open("src/pykeen/datasets/CS/"+filename+"_test.txt", "w")

for _ in train:
    trainText.write(_.replace(" ", "\t"))

for _ in valid:
    validText.write(_.replace(" ", "\t"))

for _ in test:
    testText.write(_.replace(" ", "\t"))

f.close()
trainText.close()
validText.close()
testText.close()