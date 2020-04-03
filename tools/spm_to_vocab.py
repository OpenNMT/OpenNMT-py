# converts a SentencePiece vocabulary to the format expected by dynamicdata
# (essentially converts float expected counts to "fixed precision" int pseudocounts,
# and inverts the order)
import sys
import math

OMIT = ('<unk>', '<s>', '</s>')

def convert(lines):
    for line in lines:
        w, c = line.rstrip('\n').split(None, 1)
        if w in OMIT:
            continue
        c = math.exp(float(c)) * 1000000
        c = int(c) + 1
        yield c, w 

if __name__ == '__main__':
    for c, w in convert(sys.stdin):
        print('{}\t{}'.format(c, w))
