import sys
for line_a, line_b in zip(open(sys.argv[1]).read().split('\n\n'), open(sys.argv[2]).read().split('\n\n')):
    idline = line_a.split('\n')[0].strip()
    print(idline)
    print(line_b)
    print()
