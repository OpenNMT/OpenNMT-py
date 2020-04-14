import sys

def convert(lines):
    seen_noncomment = False
    for line in lines:
        if line[0] == '#' and not seen_noncomment:
            # skip comments in beginning of file
            continue
        else:
            seen_noncomment = True
        c, w = line.rstrip('\n').split(None, 1)
        c = int(float(c)) + 1
        yield c, w 

if __name__ == '__main__':
    for c, w in convert(sys.stdin):
        print('{}\t{}'.format(c, w))
