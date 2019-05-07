'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
import random
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop,Adadelta,Adagrad,Adam

def load_noisy_data(path='mnist-with-awgn.mat'):
    # from ..utils.data_utils import get_file
    # path = get_file(path,
    #                 origin='',
    #                 file_hash='')

    import scipy.io
    f = scipy.io.loadmat(path)
    x_train, y_train = f['train_x'], f['train_y']
    x_test, y_test = f['test_x'], f['test_y']

    y_train = y_train.nonzero()[1]
    y_test = y_test.nonzero()[1]

    return (x_train, y_train), (x_test, y_test)

batch_size = 128
num_classes = 10
epochs = 20

from keras import backend as K
from keras.layers import Layer

class StochasticLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(StochasticLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # self.kernel = self.add_weight(name='kernel', 
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(StochasticLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        size = 32
        reshaped_x = tf.expand_dims(x, 2)
        x_tiled = tf.tile(reshaped_x, [1, 1, size])

        z = tf.shape(x_tiled)
        random_floats = tf.random_uniform(z)

        cast = tf.cast(x_tiled < random_floats, tf.int32)
        reduction = tf.reduce_sum(cast, 2, keepdims=True)
        reduction = tf.cast(reduction, tf.float32)
        reduction /= size

        return reduction

    def compute_output_shape(self, input_shape):
        return input_shape

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('Normal model')
model = Sequential()
model.add(Dense(512, input_shape=(784,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print('Stochastic model')
model_stochastic = Sequential()
model_stochastic.add(StochasticLayer(784, input_shape=(784,), trainable=False))
model_stochastic.add(Dense(512, input_shape=(784,), activation='relu'))
model_stochastic.add(Dropout(0.2))
model_stochastic.add(Dense(512, activation='relu'))
model_stochastic.add(Dropout(0.2))
model_stochastic.add(Dense(num_classes, activation='softmax'))

model_stochastic.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
history = model_stochastic.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model_stochastic.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

(x_train, y_train), (x_test, y_test) = load_noisy_data()
score = model.evaluate(x_test, y_test, verbose=1)
print('Normal Noisy Test loss:', score[0])
print('Normal Noisy Test accuracy:', score[1])
score = model_stochastic.evaluate(x_test, y_test, verbose=1)
print('Stochastic Noisy Test loss:', score[0])
print('Stochastic Noisy Test accuracy:', score[1])