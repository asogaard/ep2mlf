# -*- coding: utf-8 -*-
# Import(s)
import numpy as np
import keras.backend as K
from keras.layers import Layer
import tensorflow as tf
import ops

NUM_GRADIENT_REVERSALS=0
def ReverseGradient (hp_lambda):
    """
    Function factory for gradient reversal, implemented in TensorFlow.
    """

    def reverse_gradient_function (X, hp_lambda=hp_lambda):
        """Flips the sign of the incoming gradient during training."""
        global NUM_GRADIENT_REVERSALS
        grad_name = "GradientReversal{}".format(NUM_GRADIENT_REVERSALS)
        NUM_GRADIENT_REVERSALS += 1
        @tf.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * hp_lambda]

        g = K.get_session().graph
        with g.gradient_override_map({'Identity': grad_name}):
            y = tf.identity(X)
            pass

        return y

    return reverse_gradient_function


class GradientReversalLayer (Layer):

    def __init__ (self, hp_lambda, **kwargs):
        """
        ...
        """
        # Base class constructor
        super(GradientReversalLayer, self).__init__(**kwargs)

        # Member variable(s)
        self.supports_masking = False
        self.hp_lambda = hp_lambda
        self.gr_op = ReverseGradient(self.hp_lambda)
        pass

    def call (self, x, mask=None):
        return self.gr_op(x)

    def compute_output_shape (self, input_shape):
        return input_shape

    pass


class PosteriorLayer (Layer):

    def __init__ (self, nb_gmm, **kwargs):
        """
        Custom layer, modelling the posterior probability distribution for the jet mass using a gaussian mixture model (GMM)
        """
        # Base class constructor
        super(PosteriorLayer, self).__init__(**kwargs)

        # Member variable(s)
        self.nb_gmm = nb_gmm
        pass

    def call (self, x, mask=None):
        """Main call-method of the layer.
        The GMM needs to be implemented (1) within this method and (2) using
        Keras backend functions in order for the error back-propagation to work
        properly.
        """

        # Unpack list of inputs
        coeffs, means, widths, m = x

        # Compute the pdf from the GMM
        pdf = ops.GMM(m[:,0], coeffs, means, widths, self.nb_gmm)

        return K.flatten(pdf)

    def compute_output_shape (self, input_shape):
        return (input_shape[0][0], 1)

    pass
