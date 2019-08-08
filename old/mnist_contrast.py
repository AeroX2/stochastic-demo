verbose = 2
batch_size = 128
num_classes = 10
epochs = 20

from data import reduced_contrast_mnist
(x_train, y_train), (x_test, y_test) = reduced_contrast_mnist.data()

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print('Normal model')
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
model_stochastic = stochastic_model.model()
history = model_stochastic.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=verbose,
                    validation_data=(x_test, y_test))
score = model_stochastic.evaluate(x_test, y_test, verbose=verbose)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
