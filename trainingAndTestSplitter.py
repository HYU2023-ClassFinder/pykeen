f = open("tagClaims_v4.txt", "r")
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

trainText = open("train_v4.txt", "w")
validText = open("valid_v4.txt", "w")
testText = open("test_v4.txt", "w")

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