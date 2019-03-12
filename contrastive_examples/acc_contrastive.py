import sys

correct = 0
tot = 0
for normal, contrastive in zip(open(sys.argv[1]), open(sys.argv[2])):
    if len(normal.split()) > 1 or normal.strip() == '': continue
    normal = float(normal.strip())
    contrastive = float(contrastive.strip())
    if normal > contrastive:
        print(tot)
        correct += 1
    tot += 1
print(correct, tot)
print(correct/float(tot))
    
