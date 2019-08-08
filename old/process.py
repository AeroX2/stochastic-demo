import re
import sys

n = int(sys.argv[2])
with open(sys.argv[1]) as f:
    l = re.findall(r'accuracy: ([0-9.]+)',f.read())
    l = [l[i:i + n] for i in range(0, len(l), n)]
    for a in l:
        print(','.join(a))
