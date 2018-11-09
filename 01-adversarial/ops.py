# -*- coding: utf-8 -*-

# Import(s)
import numpy as np
import keras.backend as K
import tensorflow as tf


def cumulative (x):
    """
    Cumulative distribution function for the unit gaussian.
    """
    return 0.5 * (1. + tf.erf(x / np.sqrt(2.)))


def gaussian_integral_on_unit_interval (mean, width, backend=K):
    """
    Compute the integral of unit gaussians on the unit interval.

    Args:
        mean: Mean(s) of unit gaussian(s).
        width: Width(s) of unit gaussian(s).
    Returns:
        Integral of unit gaussian on [0,1]
    """
    z0 = (0. - mean) / width
    z1 = (1. - mean) / width
    integral = cumulative(z1) - cumulative(z0)
    if backend == np:
        integral = K.eval(integral)
        pass
    return integral


def gaussian (x, coeff, mean, width, backend=K):
    """
    Compute a unit gaussian using Keras-backend methods.

    Args:
        x: Variable value(s) at which to evaluate unit gaussian(s).
        coeff: Normalisation constant(s) for unit gaussian(s).
        mean: Mean(s) of unit gaussian(s).
        width: Width(s) of unit gaussian(s).
    Returns
        Function value of unit gaussian(s) evaluated at `x`.
    """
    return coeff * backend.exp( - backend.square(x - mean) / 2. / backend.square(width)) / backend.sqrt( 2. * backend.square(width) * np.pi)


def GMM (x, coeffs, means, widths, nb_gmm, backend=K):
    """
    ...
    """
    # Construct posterior p.d.f.
    pdf = backend.zeros_like(x)

    for c in range(nb_gmm):
        component  = gaussian(x, coeffs[:,c], means[:,c], widths[:,c], backend=backend)
        component /= gaussian_integral_on_unit_interval(means[:,c], widths[:,c], backend=backend)
        pdf += component
        pass

    return pdf
