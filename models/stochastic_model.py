from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop,Adadelta,Adagrad,Adam

from .stochastic_layer import StochasticLayer

def model(bit_size=None):
    model = Sequential()
    if (bit_size is None):
        model.add(StochasticLayer(784, input_shape=(784,), trainable=False))
    else:
        model.add(StochasticLayer(784, input_shape=(784,), bit_size=bit_size, trainable=False))

    model.add(Dense(512, input_shape=(784,), activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model
