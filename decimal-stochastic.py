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
  
  #sf1 = [random_() < f1 for _ in range(BIT_LENGTH_FLOAT)]
  #sf2 = [random_() < f2 for _ in range(BIT_LENGTH_FLOAT)]
  
  sd1 = [random_() < v1*2/float(RANDOM_RANGE) for _ in range(BIT_LENGTH_DECIMAL)]
  sd2 = [random_() < v2*2/float(RANDOM_RANGE) for _ in range(BIT_LENGTH_DECIMAL)]

  print(v1,v2)
  print(sum(sd1),sum(sd2))
  print(sum(sd1)/BIT_LENGTH_DECIMAL,sum(sd2)/BIT_LENGTH_DECIMAL)
  print(sum(sd1)/BIT_LENGTH_DECIMAL*RANDOM_RANGE,sum(sd2)/BIT_LENGTH_DECIMAL*RANDOM_RANGE)

  # Do addition
  #sfo = [sf1[i] | sf2[i] for i in range(BIT_LENGTH_FLOAT)]
  #carry = sum(sf1) + sum(sf2) >= BIT_LENGTH_FLOAT/2  

  rb = [random_() < 0.5 for _ in range(BIT_LENGTH_DECIMAL)]
  sdo = [sd1[i] if rb[i] else sd2[i] for i in range(BIT_LENGTH_DECIMAL)]

  #if (carry):
  #  gh = [random_() <= 1/float(RANDOM_RANGE) for _ in range(BIT_LENGTH_DECIMAL)]
  #  sdo = [sdo[i] | gh[i] for i in range(BIT_LENGTH_DECIMAL)]

  #sfa = [sf1[i] & sf2[i] for i in range(BIT_LENGTH_FLOAT)]
  #sda = [sd1[i] & sd2[i] for i in range(BIT_LENGTH_DECIMAL)]
  
  mult = 0 #sum(sda)/float(BIT_LENGTH_DECIMAL)*float(RANDOM_RANGE) + (sum(sfa)/float(BIT_LENGTH_FLOAT))
  add = sum(sdo)/float(BIT_LENGTH_DECIMAL)*float(RANDOM_RANGE)# + (sum(sfo)/float(BIT_LENGTH_FLOAT))

  return actual_mult, actual_add, mult, add

#plt.figure()

actual_mult, actual_add, stochastic_mult, stochastic_add = test()

print("Actual multiplication: ", actual_mult)
print("Actual addition: ", actual_add)

print("Stocas multiplication: ", stochastic_mult)
print("Stocas addition: ", stochastic_add)

#plt.plot(actual_mult)
#plt.plot(actual_add)
#plt.plot(stochastic_mult)
#plt.plot(stochastic_add)
#
#plt.show()
