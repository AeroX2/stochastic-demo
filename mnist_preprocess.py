from models import StochasticLayer
from data import normal_mnist
from data import noisy_mnist

import tensorflow as tf
import matplotlib.pyplot as plt

def show_image(x, y):
    a = x * 255
    a = a.astype('uint8').reshape((28,28))

    b = y * 255
    b = b.astype('uint8').reshape((28,28))

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(a, cmap='gray')
    ax2.imshow(b, cmap='gray')
    plt.show()

(x_train, y_train), (_, _) = noisy_mnist.data()

import random
random_index = random.randrange(0, x_train.shape[0])
print("Showing image for index:", random_index)
bit_size = 4

session = tf.Session()

st_layer = StochasticLayer(28*28, bit_size, invert=True)
process = st_layer.call(x_train)
result1 = session.run(process)

st_layer = StochasticLayer(28*28, bit_size, invert=False)
process = st_layer.call(x_train)
result2 = session.run(process)

st_layer = StochasticLayer(28*28, bit_size, invert=True)
process = st_layer.call(1-x_train)
result3 = session.run(process)

st_layer = StochasticLayer(28*28, bit_size, invert=False)
process = st_layer.call(1-x_train)
result4 = session.run(process)

show_image(x_train[random_index], result1[random_index])
show_image(x_train[random_index], result2[random_index])
show_image(1-x_train[random_index], result3[random_index])
show_image(1-x_train[random_index], result4[random_index])

#(x_train_n, y_train_n), (_, _) = noisy_mnist.data()


