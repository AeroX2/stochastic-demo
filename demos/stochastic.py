# -*- coding: utf-8 -*-
"""
This little code illustrates results of adding and multiplying 2 stochastic bit
streams of varying length (length default 16). The stochastic bit streams are
set by probabilities, prob1 and prob2.

The stochastic bit stream outputs are converted back to a probability for ease
of visualisation. The plots show the probabilities vary around their ideal 
value.

@author: tarah
"""
import math
import secrets
import matplotlib.pyplot as plt

class StochasticNumber:
    def __init__(self,length,prob1,prob2):
        random = secrets.SystemRandom()

        self.length = length
        self.stochnum1 = [random.random() <= prob1 for n in range(length)]
        self.stochnum2 = [random.random() <= prob2 for n in range(length)]

    def add(self):
        out = [0 for n in range(self.length)]
        sel = [secrets.randbelow(2) for n in range(self.length)] # this will (on average) give weighting of 0.5

        # Non saturating addition
        for k in range(self.length):
            # Add is a multiplexer with 2 inputs and select line that is on average 0.5
            if sel[k]:              
                out[k]=self.stochnum2[k]
            else:
                out[k]=self.stochnum1[k]

        # Saturating addition
        #for k in range(self.length):
        #    out[k] = self.stochnum1[k] | self.stochnum2[k] #Saturating addition is just logical OR

        prob_out = sum(out)/float(self.length)
        return prob_out
    
    def multiply(self):
        out = [0 for n in range(self.length)]
        for k in range(self.length):
            out[k] = (self.stochnum1[k] & self.stochnum2[k]) # multiply is just logical AND
        prob_out = sum(out)/float(self.length)
        return prob_out

class BiStochasticNumber:
    def __init__(self,length,prob1,prob2):
        random = secrets.SystemRandom()

        self.length = length
        self.stochnum1 = [random.random() <= (prob1+1)/2 for n in range(length)]
        self.stochnum2 = [random.random() <= (prob2+1)/2 for n in range(length)]

    def add(self):
        out = [0 for n in range(self.length)]
        sel = [secrets.randbelow(2) for n in range(self.length)] # this will (on average) give weighting of 0.5

        # Non saturating addition
        for k in range(self.length):
            # Add is a multiplexer with 2 inputs and select line that is on average 0.5
            if sel[k]:              
                out[k]=self.stochnum2[k]
            else:
                out[k]=self.stochnum1[k]

        # Saturating addition
        #for k in range(self.length):
        #    out[k] = self.stochnum1[k] | self.stochnum2[k] #Saturating addition is just logical OR

        prob_out = sum(out)/float(self.length)
        return (prob_out*2)-1
    
    def multiply(self):
        out = [0 for n in range(self.length)]
        for k in range(self.length):
            out[k] = not (self.stochnum1[k] ^ self.stochnum2[k]) # multiply is just logical AND

        prob_out = sum(out)/float(self.length)
        return (prob_out*2)-1

#main
import random

z = -1
prob1 = 0.5 #random.random()*2-1 #(-1 to 1)
prob2 = 0.25 #random.random()*2-1 #(-1 to 1)
#prob2 = z-prob1

resultAdd = []
resultMul = []

actualAdd = []
actualMul = []

resultAddAbs = []
resultMulAbs = []

averageMulList = []
averageAddList = []

for i in range(1,10000):
    if (i%1000==0):
        #z += 0.1
        #prob1 = random.random()*(z/2)
        #prob2 = z-prob1
        print(i)
    SN = BiStochasticNumber(i,prob1,prob2)

    add = SN.add()
    multiply = SN.multiply()

    resultAdd.append(add)
    resultMul.append(multiply)

    actualAdd.append((prob1+prob2))
    actualMul.append(prob1*prob2)

    resultAddAbs.append(abs(add-(prob1+prob2)))
    resultMulAbs.append(abs(multiply-prob1*prob2))


width = 6.4*2
height = 4.8*2

plt.figure(frameon=False, figsize=(width, height))
plt.plot(resultMul, label='Stochastic multiplication result')
plt.plot(actualMul, label='Actual multiplication result')
plt.ylim(bottom=0, top=0.35)
plt.legend()
plt.xlabel('Bit Length')
#plt.show()
plt.savefig('Figure_1_bi.png')

#plt.figure(frameon=False, figsize=(width, height))
#plt.plot(resultAdd, label='Stochastic addition result')
#plt.plot(actualAdd, label='Actual addition result')
#plt.legend()
##plt.xlabel('')
#plt.savefig('Figure_3_sss.png')

plt.figure(frameon=False, figsize=(width, height))
plt.ylim(top=0.2)
plt.bar([i for i in range(1,10000)], resultMulAbs, label='Multiplication Absolute Difference') #0.5x025 = 0.125 is the expected output for prob1 = 0.5 and prob2 = 0.25
plt.legend()
plt.xlabel('Bit Length')
plt.savefig('Figure_1_bi_abs.png')

#plt.figure(frameon=False, figsize=(width, height))
#plt.ylim(top=0.2)
#plt.bar([i for i in range(1,iters)], resultAddAbs, label='Addition Absolute Difference') #0.5x025 = 0.125 is the expected output for prob1 = 0.5 and prob2 = 0.25
#plt.legend()
#plt.xlabel('Bit Length')
#plt.savefig('Figure_2-abs.png')
