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

        self.stochnum1 = [1 if random.random() <= prob1 else 0 for n in range(length)]
        self.stochnum2 = [1 if random.random() <= prob2 else 0 for n in range(length)]

        # self.stochnum1 = [1]*math.ceil(length*prob1)
        # self.stochnum1 += [0]*math.ceil(length*(1-prob1))
        # random.shuffle(self.stochnum1)
        # self.stochnum2 = [1]*math.ceil(length*prob2)
        # self.stochnum2 += [0]*math.ceil(length*(1-prob2))
        # random.shuffle(self.stochnum2)

        # print(self.stochnum1)
        # print(self.stochnum2)

        #This doesn't seem correct
        #self.stochnum1 = [0 for n in range(length)]
        #for n in range(int(length*prob1)):
        #    self.stochnum1[secrets.randbelow(length)] = 1

        #self.stochnum2 = [0 for n in range(length)]            
        #for n in range(int(length*prob2)):
        #    self.stochnum2[secrets.randbelow(length)] = 1
            
        self.length = length
        self.out = [0 for n in range(length)]
        self.probOut = 0.0

    def add(self):
        self.Sel = [secrets.randbelow(2) for n in range(self.length)] # this will (on average) give weighting of 0.5

        #for k in range(self.length):
        #    if self.Sel[k]==0:              #Add is a multiplexer with 2 inputs and select line that is on average 0.5
        #        self.out[k]=self.stochnum1[k]
        #    else:
        #        self.out[k]=self.stochnum2[k]

        for k in range(self.length):
            self.out[k] = self.stochnum1[k] | self.stochnum2[k] #saturating addition is just logical OR

        probOut = float(sum(self.out))/float(self.length)
        return probOut
    
    def multiply(self):
        for k in range(self.length):
            self.out[k] = (self.stochnum1[k] & self.stochnum2[k]) #multiply is just logical AND
        self.probOut = float(sum(self.out))/float(self.length)
        return self.probOut

#main
import random
iters = 1000

z = 0.1
prob1 = random.random()*(z/2)
prob2 = z-prob1

print(prob1+prob2)

resultAdd = []
resultMul = []

actualAdd = []
actualMul = []

resultAddAbs = []
resultMulAbs = []

averageMulList = []
averageAddList = []
ROLLING = 30

for i in range(1,10000):
    if (i%1000==0):
        z += 0.1
        prob1 = random.random()*(z/2)
        prob2 = z-prob1
        print(i)
    SN = StochasticNumber(1000,prob1,prob2)

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

#plt.figure(frameon=False, figsize=(width, height))
#plt.plot(resultMul, label='Stochastic multiplication result')
#plt.plot(actualMul, label='Actual multiplication result')
#plt.legend()
#plt.xlabel('Bit Length')
#plt.savefig('Figure_1.png')
#
plt.figure(frameon=False, figsize=(width, height))
plt.plot(resultAdd, label='Stochastic addition result')
plt.plot(actualAdd, label='Actual addition result')
plt.legend()
#plt.xlabel('')
plt.savefig('Figure_3_sss.png')
#
#plt.figure(frameon=False, figsize=(width, height))
#plt.ylim(top=0.2)
#plt.bar([i for i in range(1,iters)], resultMulAbs, label='Multiplication Absolute Difference') #0.5x025 = 0.125 is the expected output for prob1 = 0.5 and prob2 = 0.25
#plt.legend()
#plt.xlabel('Bit Length')
#plt.savefig('Figure_1-abs.png')
#
#plt.figure(frameon=False, figsize=(width, height))
#plt.ylim(top=0.2)
#plt.bar([i for i in range(1,iters)], resultAddAbs, label='Addition Absolute Difference') #0.5x025 = 0.125 is the expected output for prob1 = 0.5 and prob2 = 0.25
#plt.legend()
#plt.xlabel('Bit Length')
#plt.savefig('Figure_2-abs.png')
