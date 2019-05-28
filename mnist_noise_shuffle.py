verbose = 2
batch_size = 128
num_classes = 10
epochs = 20

from data import normal_mnist,noisy_mnist
(x_train_m, y_train_m), (x_test_m, y_test_m) = normal_mnist.data()
(x_train_n, y_train_n), (x_test_n, y_test_n) = noisy_mnist.data()

import numpy as np
x_train = np.concatenate((x_train_m, x_train_n))
y_train = np.concatenate((y_train_m, y_train_n))
x_test = np.concatenate((x_test_m, x_test_n))
y_test = np.concatenate((y_test_m, y_test_n))

def shuffle(a,b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

shuffle(x_train, y_train)
shuffle(x_test, y_test)

from models import normal_model
model = normal_model.model()
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=verbose,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=verbose)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print('Stochastic model')
from models import stochastic_model
model = stochastic_model.model()
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=verbose,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=verbose)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
