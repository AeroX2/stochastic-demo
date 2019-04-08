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
import secrets
import matplotlib.pyplot as plt

class StochasticNumber:
    def __init__(self,length,prob1,prob2):
        random = secrets.SystemRandom()

        self.stochnum1 = [1 if random.random() <= prob1 else 0 for n in range(length)]
        self.stochnum2 = [1 if random.random() <= prob2 else 0 for n in range(length)]

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
        #print(self.Sel)
        for k in range(self.length):
            if self.Sel[k]==0:              #Add is a multiplexer with 2 inputs and select line that is on average 0.5
                self.out[k]=self.stochnum1[k]
            else:
                self.out[k]=self.stochnum2[k]

        probOut = float(sum(self.out))/float(self.length)
        return probOut
    
    def multiply(self):
        for k in range(self.length):
            self.out[k] = self.stochnum1[k] & self.stochnum2[k] #multiply is just logical AND
        self.probOut = float(sum(self.out))/float(self.length)
        return self.probOut

#main
length = 100
iters = 1000
prob1 = 0.5
prob2 = 0.25

resultAdd = []
resultMult = []
actualAdd = []
actualMul = []

averageMulList = []
averageAddList = []
ROLLING = 30

for i in range(iters):
    SN = StochasticNumber(length,prob1,prob2)
    resultAdd.append(SN.add())
    resultMult.append(SN.multiply())
    actualAdd.append(0.5*(prob1+prob2))
    actualMul.append(prob1*prob2)

    if (len(resultMult) > ROLLING):
        average = sum(resultMult[i-ROLLING:i])
        averageMulList.append(average / ROLLING)
    else:
        averageMulList.append(0)

    if (len(resultAdd) > ROLLING):
        average = sum(resultAdd[i-ROLLING:i])
        averageAddList.append(average / ROLLING)
    else:
        averageAddList.append(0)
    

plt.figure()
plt.plot(resultAdd) #0.5 x (0.5+0.25) = 0.375 is the expected output for prob1 = 0.5 and prob2 = 0.25
plt.plot(resultMult) #0.5x025 = 0.125 is the expected output for prob1 = 0.5 and prob2 = 0.25
plt.plot(actualAdd)
plt.plot(actualMul)
plt.plot(averageMulList)
plt.plot(averageAddList)
plt.show()
                    
