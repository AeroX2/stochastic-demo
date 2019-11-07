# -*- coding: utf-8 -*-
"""
This little code illustrates results of adding and multiplying 2 stochastic bit
streams of varying length (length default 16). The stochastic bit streams are
set by probabilities, prob1 and prob2.

The stochastic bit stream outputs are converted back to a probability for ease
of visualisation. The plots show the probabilities vary around their ideal 
value.

@author: Tarah, James
"""
import math
import random
import secrets
import matplotlib.pyplot as plt

class StochasticNumber:
    def __init__(self,length,prob1,prob2):
        random = secrets.SystemRandom()

        self.length = length
        self.stochnum1 = [random.random() <= prob1 for n in range(length)]
        self.stochnum2 = [random.random() <= prob2 for n in range(length)]

    def add(self, saturating=False):
        out = [0 for n in range(self.length)]
        sel = [secrets.randbelow(2) for n in range(self.length)] # this will (on average) give weighting of 0.5

        if (saturating):
            # Saturating addition
            for k in range(self.length):
                out[k] = self.stochnum1[k] | self.stochnum2[k] #Saturating addition is just logical OR
        else:
            # Non saturating addition
            for k in range(self.length):
                # Add is a multiplexer with 2 inputs and select line that is on average 0.5
                if sel[k]:              
                    out[k]=self.stochnum2[k]
                else:
                    out[k]=self.stochnum1[k]

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

    def add(self, saturating=False):
        out = [0 for n in range(self.length)]
        sel = [secrets.randbelow(2) for n in range(self.length)] # this will (on average) give weighting of 0.5

        if (saturating):
            # Saturating addition
            for k in range(self.length):
                out[k] = self.stochnum1[k] | self.stochnum2[k] #Saturating addition is just logical OR
        else:
            # Non saturating addition
            for k in range(self.length):
                # Add is a multiplexer with 2 inputs and select line that is on average 0.5
                if sel[k]:              
                    out[k]=self.stochnum2[k]
                else:
                    out[k]=self.stochnum1[k]

        prob_out = (sum(out) + -1*(self.length-sum(out)))/float(self.length)
        return (prob_out) #*2)-1
    
    def multiply(self):
        out = [0 for n in range(self.length)]
        for k in range(self.length):
            out[k] = not (self.stochnum1[k] ^ self.stochnum2[k]) # multiply is just logical AND

        prob_out = (sum(out) + -1*(self.length-sum(out)))/float(self.length)
        return (prob_out) #*2)-1

iters = 10000

add_abs = []
mul_abs = []

for bit_length in range(1,iters):
    if (bit_length%1000==0):
        print(bit_length)

    add_abs.append(0)
    mul_abs.append(0)

    for ii in range(10):
        prob1 = random.random()
        prob2 = random.random()

        #SN = StochasticNumber(bit_length,prob1,prob2)
        SN = BiStochasticNumber(bit_length,prob1,prob2)

        add = SN.add()
        multiply = SN.multiply()

        add_abs[-1] += abs(add*2-(prob1+prob2))
        mul_abs[-1] += abs(multiply-(prob1*prob2))

for i in range(len(add_abs)):
    add_abs[i] /= 10
    mul_abs[i] /= 10

width = 6.4*2
height = 4.8*2

plt.figure(frameon=False, figsize=(width, height))
plt.ylim(top=0.125)
plt.bar([i for i in range(1,iters)], mul_abs, label='Multiplication Absolute Difference', width=1.0)
plt.legend()
plt.xlabel('Bit Length')
plt.ylabel('Value')
#plt.show()
plt.savefig('mult-bi-abs.png')

plt.figure(frameon=False, figsize=(width, height))
plt.ylim(top=0.35)
plt.bar([i for i in range(1,iters)], add_abs, label='Addition Absolute Difference', width=1.0)
plt.legend()
plt.xlabel('Bit Length')
plt.ylabel('Value')
#plt.show()
plt.savefig('add-bi-abs.png')

import pickle
with open('mult-bi', 'wb') as fp:
    pickle.dump(mul_abs, fp)

with open('add-bi', 'wb') as fp:
    pickle.dump(add_abs, fp)
