from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop,Adadelta,Adagrad,Adam

from .stochastic_layer import StochasticLayer

def model():
    model = Sequential()
    model.add(StochasticLayer(784, input_shape=(784,), trainable=False))
    model.add(Dense(512, input_shape=(784,), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    return model
