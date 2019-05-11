verbose = 2
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
model.add(Conv2D(28, kernel_size=(3,3), input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
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
model_stochastic.add(Conv2D(28, kernel_size=(3,3), input_shape=(28,28,1)))
model_stochastic.add(MaxPooling2D(pool_size=(2,2)))
model_stochastic.add(Flatten())
model_stochastic.add(Dense(128, activation='relu'))
model_stochastic.add(Dropout(0.2))
model_stochastic.add(Dense(10,activation='softmax'))

model_stochastic.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
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
(x_train, y_train), (x_test, y_test) = noisy_mnist.data()
score = model.evaluate(x_test, y_test, verbose=verbose)
print('Conv2D Noisy Test loss:', score[0])
print('Conv2D Noisy Test accuracy:', score[1])
score = model_stochastic.evaluate(x_test, y_test, verbose=verbose)
print('Stochastic Conv2D Noisy Test loss:', score[0])
print('Stochastic Conv2D Noisy Test accuracy:', score[1])
