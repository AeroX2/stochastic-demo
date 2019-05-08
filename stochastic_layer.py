import tensorflow as tf
from keras import backend as K
from keras.layers import Layer

class StochasticLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(StochasticLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # self.kernel = self.add_weight(name='kernel', 
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(StochasticLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        size = 32
        reshaped_x = tf.expand_dims(x, 2)
        x_tiled = tf.tile(reshaped_x, [1, 1, size])

        z = tf.shape(x_tiled)
        random_floats = tf.random_uniform(z)

        cast = tf.cast(x_tiled < random_floats, tf.int32)
        reduction = tf.reduce_sum(cast, 2, keepdims=True)
        reduction = tf.cast(reduction, tf.float32)
        reduction /= size

        return reduction

    def compute_output_shape(self, input_shape):
        return input_shape