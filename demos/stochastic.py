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

prob1 = 0.5  #random.random()*2-1 #(-1 to 1)
prob2 = 0.25 #random.random()*2-1 #(-1 to 1)

result_add = []
result_mul = []

actual_add = []
actual_mul = []

for bit_length in range(1,10000):
    if (bit_length%1000==0):
        print(bit_length)
    #SN = StochasticNumber(bit_length,prob1,prob2)
    SN = BiStochasticNumber(bit_length,prob1,prob2)

    add = SN.add()
    multiply = SN.multiply()

    result_add.append(add*2)
    result_mul.append(multiply)

    actual_add.append((prob1+prob2))
    actual_mul.append(prob1*prob2)


width = 6.4*2
height = 4.8*2
plt.rcParams.update({'font.size': 22})

plt.figure(frameon=False, figsize=(width, height))
plt.plot(result_mul, label='Stochastic multiplication result')
plt.plot(actual_mul, label='Actual multiplication result')
plt.ylim(bottom=0, top=0.35)
plt.legend()
plt.xlabel('Bit Length')
plt.ylabel('Value')
#plt.show()
plt.savefig('mult-bi.png')

plt.figure(frameon=False, figsize=(width, height))
plt.plot(result_add, label='Stochastic addition result')
plt.plot(actual_add, label='Actual addition result')
plt.legend()
plt.xlabel('Bit Length')
plt.ylabel('Value')
#plt.show()
plt.savefig('add-bi.png')

