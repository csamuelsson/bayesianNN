# Dependencies
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import json

# Import hyperparams from JSON file
with open('hyperparams.json') as json_data:
    hyperparams = json.load(json_data)
    json_data.close()

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
  dist = tfd.Normal(loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(hyperparams['prior_beliefs']['prior_distribution_std']))
  batch_ndims = tf.size(dist.batch_shape_tensor())
  return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)