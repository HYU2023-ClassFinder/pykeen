f = open("tagClaims_vzscoreUpper40.txt", "r")

token = []

for line in f:
    token.append(line.split()[0])
    token.append(line.split()[2])

token = list(set(token))

print(token)

f.close()