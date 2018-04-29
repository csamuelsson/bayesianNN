# Dependencies
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from flags import *

FLAGS = flags.FLAGS

tfd = tf.contrib.distributions

def default_multivariate_normal_fn(dtype, shape, name, trainable,
                                   add_variable_fn):
  """Creates multivariate standard `Normal` distribution.
  Args:
    dtype: Type of parameter's event.
    shape: Python `list`-like representing the parameter's event shape.
    name: Python `str` name prepended to any created (or existing)
      `tf.Variable`s.
    trainable: Python `bool` indicating all created `tf.Variable`s should be
      added to the graph collection `GraphKeys.TRAINABLE_VARIABLES`.
    add_variable_fn: `tf.get_variable`-like `callable` used to create (or
      access existing) `tf.Variable`s.
  Returns:
    Multivariate standard `Normal` distribution.
  """
  del name, trainable, add_variable_fn   # unused
  dist = tfd.Normal(loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(FLAGS.prior_sigma_1))
  batch_ndims = tf.size(dist.batch_shape_tensor())
  return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)


def make_scale_mixture_prior_fn(dtype, shape, name, trainable,
                                   add_variable_fn):
    """Creates multivariate `Mixture` distribution
    Args:
        dtype: Type of parameter's event
        shape: Python `list`-like representing the parameter's event shape.
        name: Python `str` name prepended to any created (or existing)
            `tf.Variable`s.
        trainable: Python `bool` indicating all created `tf.Variable`s should be
            added to the graph collection `GraphKeys.TRAINABLE_VARIABLES`.
        add_variable_fn: `tf.get_variable`-like `callable` used to create (or
        access existing) `tf.Variable`s.
    Returns:
        Multivariate scale `Mixture` distribution.
    """
    del name, trainable, add_variable_fn # not used
    '''dist = tfd.MixtureSameFamily(
        cat=tfd.Categorical(probs=[FLAGS.prior_pi, 1.-FLAGS.prior_pi], validate_args=True),
        components=[
            tfd.Normal(loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(FLAGS.prior_sigma_1)),
            tfd.Normal(loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(FLAGS.prior_sigma_2)),
        ])'''
    dist = tfd.MixtureSameFamily(
          mixture_distribution=tfd.Categorical(probs=[FLAGS.prior_pi, 1-FLAGS.prior_pi]),
          components_distribution=tfd.MultivariateNormalDiag(
              loc=[tf.zeros(shape, dtype), tf.zeros(shape, dtype)], scale_identity_multiplier=[dtype.as_numpy_dtype(FLAGS.prior_sigma_1), dtype.as_numpy_dtype(FLAGS.prior_sigma_2)]))

    # set dimensions for dist
    # dist.shape([728, 128])

    batch_ndims = tf.size(dist.batch_shape_tensor())
    return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
