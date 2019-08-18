verbose = 2
batch_size = 128
num_classes = 10
epochs = 20

import tensorflow as tf
print("GPU Available: ", tf.test.is_gpu_available())

from data import normal_mnist
(x_train, y_train), (x_test, y_test) = normal_mnist.data()

#Setup
#from models import normal_model
#model = normal_model.model()
#history = model.fit(x_train, y_train,
#                    batch_size=batch_size,
#                    epochs=epochs,
#                    verbose=verbose,
#                    validation_data=(x_test, y_test))

from models import stochastic_model
model_stochastic = stochastic_model.model()
history = model_stochastic.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=verbose,
                    validation_data=(x_test, y_test))

#Normal test
#print('Normal model')
#score = model.evaluate(x_test, y_test, verbose=verbose)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

print('Stochastic model')
score = model_stochastic.evaluate(x_test, y_test, verbose=verbose)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#Noisy test
from data import noisy_mnist
(_, _), (x_test, y_test) = noisy_mnist.data()
score = model.evaluate(x_test, y_test, verbose=verbose)
print('Normal Noisy Test loss:', score[0])
print('Normal Noisy Test accuracy:', score[1])

score = model_stochastic.evaluate(x_test, y_test, verbose=verbose)
print('Stochastic Noisy Test loss:', score[0])
print('Stochastic Noisy Test accuracy:', score[1])


#Custom test
from data import custom_mnist
(_, _), (x_test, y_test) = custom_mnist.data()

score = model.evaluate(x_test, y_test, verbose=verbose)
print('Normal Custom Test loss:', score[0])
print('Normal Custom Test accuracy:', score[1])

score = model_stochastic.evaluate(x_test, y_test, verbose=verbose)
print('Stochastic Custom Test loss:', score[0])
print('Stochastic Custom Test accuracy:', score[1])
