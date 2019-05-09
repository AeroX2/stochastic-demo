#!/bin/env python3
import os
import sys

from subprocess import Popen, PIPE, CalledProcessError

if (len(sys.argv) <= 1):
    print("benchmark.py file [amount]")
elif (len(sys.argv) <= 2):
    print("No amount given, using default value of 20")
    amount = 20
else:
    amount = int(sys.argv[2])

f = sys.argv[1]
fs = os.path.basename(f)
f_name, f_ext = os.path.splitext(fs)

f_temp = f_name+'.temp.txt'
f_final = f_name+'.txt'

#TODO: Maybe rewrite these as native python :P
cmd = "echo ================ > {}".format(f_temp)
os.system(cmd)
cmd = "echo Start of new run >> {}".format(f_temp)
os.system(cmd)
cmd = "echo ================ >> {}".format(f_temp)
os.system(cmd)

for i in range(amount):
    print("Iteration: {}".format(i+1))
    cmd = "python3 {} {} | tee -a {}".format(f, i, f_temp)
    with Popen(cmd, stdout=PIPE, bufsize=1, universal_newlines=True, shell=True) as p:
        for line in p.stdout:
            print(line, end='')

cmd = "cat {} >> results/{}".format(f_temp, f_final)
os.system(cmd)

cmd = "rm {}".format(f_temp)
os.system(cmd)
