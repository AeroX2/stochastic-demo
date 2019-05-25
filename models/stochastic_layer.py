import tensorflow as tf
from keras import backend as K
from keras.layers import Layer

class StochasticLayer(Layer):

    def __init__(self, output_dim, bit_size = 8, invert = False, **kwargs):
        print("Stochastic layer using a bit size of {}".format(bit_size))
        self.output_dim = output_dim
        self.bit_size = bit_size
        self.invert = invert
        super(StochasticLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # self.kernel = self.add_weight(name='kernel', 
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(StochasticLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        reshaped_x = tf.expand_dims(x, 2)
        x_tiled = tf.tile(reshaped_x, [1, 1, self.bit_size])

        z = tf.shape(x_tiled)
        random_floats = tf.random_uniform(z)

        if (self.invert):
            cast = tf.cast(x_tiled <= random_floats, tf.int8)
        else:
            cast = tf.cast(random_floats <= x_tiled, tf.int8)

        reduction = tf.reduce_sum(cast, 2, keepdims=True)
        reduction = tf.cast(reduction, tf.float32)
        reduction /= self.bit_size

        return reduction

    def compute_output_shape(self, input_shape):
        return input_shape
