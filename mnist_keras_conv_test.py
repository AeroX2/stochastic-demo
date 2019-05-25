verbose = 1
batch_size = 128
num_classes = 10
epochs = 20

from data import normal_mnist
(x_train, y_train), (x_test, y_test) = normal_mnist.data()

from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten,Reshape
from keras.optimizers import RMSprop,Adadelta,Adagrad,Adam

model = Sequential()
model.add(Reshape((28,28,1)))
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=verbose,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=verbose)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

from models import StochasticLayer

model_stochastic = Sequential()
model_stochastic.add(StochasticLayer(784, input_shape=(784,), trainable=False))
model_stochastic.add(Reshape((28,28,1)))
model_stochastic.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model_stochastic.add(MaxPooling2D((2, 2)))
model_stochastic.add(Conv2D(64, (3, 3), activation='relu'))
model_stochastic.add(MaxPooling2D((2, 2)))
model_stochastic.add(Conv2D(64, (3, 3), activation='relu'))
model_stochastic.add(Flatten())
model_stochastic.add(Dense(64, activation='relu'))
model_stochastic.add(Dense(10, activation='softmax'))

model_stochastic.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
history = model_stochastic.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=verbose,
                    validation_data=(x_test, y_test))
score = model_stochastic.evaluate(x_test, y_test, verbose=verbose)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

from data import noisy_mnist
(_, _), (x_test, y_test) = noisy_mnist.data()
score = model.evaluate(x_test, y_test, verbose=verbose)
print('Conv2D Noisy Test loss:', score[0])
print('Conv2D Noisy Test accuracy:', score[1])
for _ in range(10):
    score = model_stochastic.evaluate(x_test, y_test, verbose=verbose)
    print('Stochastic Conv2D Noisy Test loss:', score[0])
    print('Stochastic Conv2D Noisy Test accuracy:', score[1])

from data import custom_mnist
(_, _), (x_test, y_test) = custom_mnist.data()
score = model.evaluate(x_test, y_test, verbose=verbose)
print('Conv2D Custom Test loss:', score[0])
print('Conv2D Custom Test accuracy:', score[1])
for _ in range(10):
    score = model_stochastic.evaluate(x_test, y_test, verbose=verbose)
    print('Stochastic Conv2D Custom Test loss:', score[0])
    print('Stochastic Conv2D Custom Test accuracy:', score[1])
