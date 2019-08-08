verbose = 2
batch_size = 128
epochs = 20

from data import normal_mnist
(x_train, y_train), (x_test, y_test) = normal_mnist.data()

try:
    import sys
    bitsize = 2**int(sys.argv[1])
except:
    bitsize = 32

print('Stochastic model')
from models import stochastic_model
model = stochastic_model.model(bit_size=bitsize)
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=verbose,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=verbose)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
