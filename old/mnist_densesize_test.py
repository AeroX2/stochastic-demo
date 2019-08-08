verbose = 2
batch_size = 128
epochs = 20

from data import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.data()

try:
    import sys
    densesize = 2**(int(sys.argv[1])+1)
except:
    densesize = 512

print('Normal model')
from models import normal_model
model = normal_model.model(dense_size=densesize)
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
model = stochastic_model.model(dense_size=densesize)
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=verbose,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=verbose)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
