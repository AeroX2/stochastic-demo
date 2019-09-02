import keras
from keras.datasets import mnist

def data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    #from keras.utils import to_categorical
    #y_train = to_categorical(y_train, 10)
    #y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)
