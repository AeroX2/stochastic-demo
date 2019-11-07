import math
import random
import secrets
import matplotlib.pyplot as plt

RANDOM_RANGE = 20
BIT_LENGTH_DECIMAL = 1000
BIT_LENGTH_FLOAT = 1000

def test():
  # The 2 numbers we are planning to add/mult
  v1 = random.uniform(0, RANDOM_RANGE/2);
  v2 = random.uniform(0, RANDOM_RANGE/2);
  
  actual_mult = v1*v2
  actual_add = v1+v2
  
  f1,d1 = math.modf(v1)
  f2,d2 = math.modf(v2)

  random_ = lambda: random.random()
  
  sf1 = [random_() < f1 for _ in range(BIT_LENGTH_FLOAT)]
  sf2 = [random_() < f2 for _ in range(BIT_LENGTH_FLOAT)]
  
  sd1 = [random_() < d1/float(RANDOM_RANGE) for _ in range(BIT_LENGTH_DECIMAL)]
  sd2 = [random_() < d2/float(RANDOM_RANGE) for _ in range(BIT_LENGTH_DECIMAL)]

  # Do addition
  rb = [random_() < 0.5 for _ in range(BIT_LENGTH_DECIMAL)]
  sfo = [sf1[i] if rb[i] else sf2[i] for i in range(BIT_LENGTH_FLOAT)]
  carry = sum(sf1) + sum(sf2) >= BIT_LENGTH_FLOAT/2  

  rb = [random_() < 0.5 for _ in range(BIT_LENGTH_DECIMAL)]
  sdo = [sd1[i] if rb[i] else sd2[i] for i in range(BIT_LENGTH_DECIMAL)]

  #if (carry):
  #  gh = [random_() <= 1/float(RANDOM_RANGE) for _ in range(BIT_LENGTH_DECIMAL)]
  #  rb = [random_() < 0.5 for _ in range(BIT_LENGTH_DECIMAL)]
  #  sdo = [sdo[i] if rb[i] else gh[i] for i in range(BIT_LENGTH_DECIMAL)]

  sfa = [sf1[i] & sf2[i] for i in range(BIT_LENGTH_FLOAT)]
  sda = [sd1[i] & sd2[i] for i in range(BIT_LENGTH_DECIMAL)]
  
  mult = (sum(sda)/float(BIT_LENGTH_DECIMAL)*float(RANDOM_RANGE**2)) + (sum(sfa)/float(BIT_LENGTH_FLOAT))
  add = (sum(sdo)/float(BIT_LENGTH_DECIMAL)*float(RANDOM_RANGE)) + (sum(sfo)/float(BIT_LENGTH_FLOAT))

  sd1 = [random_() <= v1/float(RANDOM_RANGE) for _ in range(BIT_LENGTH_DECIMAL+BIT_LENGTH_FLOAT)]
  sd2 = [random_() <= v2/float(RANDOM_RANGE) for _ in range(BIT_LENGTH_DECIMAL+BIT_LENGTH_FLOAT)]

  rb = [random_() < 0.5 for _ in range(BIT_LENGTH_DECIMAL+BIT_LENGTH_FLOAT)]
  sdo = [sd1[i] if rb[i] else sd2[i] for i in range(BIT_LENGTH_DECIMAL+BIT_LENGTH_FLOAT)]
  #print(sum(sdo))
  #print(sum(sdo)/float(BIT_LENGTH_DECIMAL+BIT_LENGTH_FLOAT)*float(RANDOM_RANGE))

  sda = [sd1[i] & sd2[i] for i in range(BIT_LENGTH_DECIMAL+BIT_LENGTH_FLOAT)]

  nmult = (sum(sda)/float(BIT_LENGTH_DECIMAL+BIT_LENGTH_FLOAT))*float(RANDOM_RANGE)
  nadd = (sum(sdo)/float(BIT_LENGTH_DECIMAL+BIT_LENGTH_FLOAT))*float(RANDOM_RANGE)

  return actual_mult, actual_add, mult, add*2, nmult*RANDOM_RANGE, nadd*2

#plt.figure()

aa = []
ast = []
anst = []

ma = []
mst = []
mnst = []

for i in range(10000):
  actual_mult, actual_add, stochastic_mult, stochastic_add, nstochastic_mult, nstochastic_add = test()

  if (i%1000==0):
    print(i)

  #aa.append(actual_add)
  ast.append(abs(actual_add-stochastic_add))
  anst.append(abs(actual_add-nstochastic_add))

  #ma.append(actual_mult)
  mst.append(abs(actual_mult-stochastic_mult))
  mnst.append(abs(actual_mult-nstochastic_mult))

#print("Actual multiplication: ", actual_mult)
#print("Actual addition: ", actual_add)

#print("Stocas multiplication: ", stochastic_mult)
#print("Normal Stocas addition: ", nstochastic_add*2)
#print("Stocas addition: ", stochastic_add*2)

width = 6.4*2
height = 4.8*2

plt.figure(frameon=False, figsize=(width, height))
plt.bar([i for i in range(10000)], ast, label='Split addition stochastic result', width=1.0)
plt.bar([i for i in range(10000)], anst, label='Normal addition stochastic result', width=1.0)
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.legend()
plt.savefig('split-add.png')

plt.figure(frameon=False, figsize=(width, height))
plt.bar([i for i in range(10000)], mst, label='Split multiplication stochastic result', width=1.0)
plt.bar([i for i in range(10000)], mnst, label='Normal multiplication stochastic result', width=1.0)
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.legend()
plt.savefig('split-mult.png')
