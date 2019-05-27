from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop,Adadelta,Adagrad,Adam

def model(dense_size=512):
    print("Using a dense size of {}".format(dense_size))

    model = Sequential()
    model.add(Dense(dense_size, input_shape=(784,), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(dense_size, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model
