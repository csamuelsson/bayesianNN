"""Bayesian neural network using variational inference
(see, e.g., Blundell et al. (2015); Kucukelbir et al. (2016)).

Inspired by autograd's Bayesian neural network example.
This example prettifies some of the tensor naming for visualization in
TensorBoard. To view TensorBoard, run `tensorboard --logdir=log`.

References
----------
http://edwardlib.org/tutorials/bayesian-neural-network
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf
import pandas as pd

from edward.models import Normal

tf.flags.DEFINE_integer("N", default=40, help="Number of data points.")
tf.flags.DEFINE_integer("D", default=1, help="Number of features.")
tf.flags.DEFINE_string("activation", default="relu", help="Activation function to be applied for the layers")
tf.flags.DEFINE_integer("num_hidden_layers", default=1, help="Number of hidden layers")
tf.flags.DEFINE_integer("hidden_layers_dim", default=50, help="Number of neurons in each hidden layer")

FLAGS = tf.flags.FLAGS
FLAGS.D = 6849
FLAGS.N = 600
FLAGS.hidden_layers_dim = 100
FLAGS.num_hidden_layers = 1

# LOAD MEDTECH DATASET
data = pd.read_csv("variables.csv.txt", sep='\t')

# last column is the response variable
y = np.array(data[data.columns[-1]], dtype='float32')
# rest of the variables (except the first) are the features
x = np.array(data[data.columns[1:-1]], dtype='float32')

# since x_6850 -> x_6908 has variance 0 they don't contribute to our mapping, therefore we remove them here
x = x[0:len(x), 0:6849]

# shuffle the data
data = data.sample(frac=1)

# amount of training
train_range = int(0.8*len(x))

x_train = x[:train_range]
y_train = y[:train_range]
y_test = y[train_range:]
x_test = x[train_range:]

# normalize X, note that we are standardizing the test data with respect to the test mean and std 

x_train = (x_train - np.mean(x_train, axis=0))/np.std(x_train, axis=0)
x_test = (x_test - np.mean(x_train, axis=0))/np.std(x_train, axis=0)

mean_train = np.mean(y_train)
mean_test = np.mean(y_test)

def main(_):
  def neural_network(X):
    # first layers - determined by input dataset, testing for relu or sigmoid as activation
    if FLAGS.activation == "relu":
      h = tf.nn.relu(tf.matmul(X, W_0) + b_0)
    else:
      h = tf.sigmoid(tf.matmul(X, W_0) + b_0)
    
    # hidden layers - assumed to be symmetrical in terms of their hyperparameters
    for i in range(FLAGS.num_hidden_layers):
      # test for activation function
      if FLAGS.activation == "relu":
        h = tf.nn.relu(tf.matmul(h, W_i) + b_i)
      else:
        h = tf.sigmoid(tf.matmul(h, W_i) + b_i)
    
    # add last layer, no activation function since unbounded outcome
    h = tf.matmul(h, W_last) + b_last
    return tf.reshape(h, [-1])

    ''' Previous working example:
    h = tf.nn.relu(tf.matmul(X, W_0) + b_0)
    h = tf.nn.relu(tf.matmul(h, W_1) + b_1)
    h = tf.matmul(h, W_2) + b_2
    return tf.reshape(h, [-1])'''

  ed.set_seed(42)

  # DATA
  # X_train, y_train = build_toy_dataset(FLAGS.N)

  # MODEL
  with tf.name_scope("model"):
    model_dict = {}
    W_0 = Normal(loc=tf.zeros([FLAGS.D, 10]), scale=tf.ones([FLAGS.D, 10]), name="W_0")
    b_0 = Normal(loc=tf.zeros(10), scale=tf.ones(10), name="b_0")
    for i in range(1, FLAGS.num_hidden_layers):
      model_dict["W_" + str(i)] = Normal(loc=tf.zeros([10, 10]), scale=tf.ones([10, 10]), name="W_i")
      model_dict["b_" + str(i)] = Normal(loc=tf.zeros(10), scale=tf.ones(10), name="b_i")

    W_last = Normal(loc=tf.zeros([10, 1]), scale=tf.ones([10, 1]), name="W_last")
    b_last = Normal(loc=tf.zeros(1), scale=tf.ones(1), name="b_last")

    X = tf.placeholder(tf.float32, [FLAGS.N, FLAGS.D], name="X")
    y = Normal(loc=neural_network(X), scale=0.1 * tf.ones(FLAGS.N), name="y")

  # INFERENCE
  with tf.variable_scope("posterior"):
    with tf.variable_scope("qW_0"):
        loc = tf.get_variable("loc", [FLAGS.D, FLAGS.hidden_layers_dim])
        scale = tf.nn.softplus(tf.get_variable("scale", [FLAGS.D, FLAGS.hidden_layers_dim]))
        qW_0 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qb_0"):
        loc = tf.get_variable("loc", [FLAGS.hidden_layers_dim])
        scale = tf.nn.softplus(tf.get_variable("scale", [FLAGS.hidden_layers_dim]))
        qb_0 = Normal(loc=loc, scale=scale)

    for i in range(1, FLAGS.num_hidden_layers):
        with tf.variable_scope("qW_" + str(i)):
            loc = tf.get_variable("loc", [FLAGS.hidden_layers_dim, FLAGS.hidden_layers_dim])
            scale = tf.nn.softplus(tf.get_variable("scale", [FLAGS.hidden_layers_dim, FLAGS.hidden_layers_dim]))
            tf.get_variable(Normal(loc=loc, scale=scale), name="qW_"+str(i))
        with tf.variable_scope("qb_" + str(i)):
            loc = tf.get_variable("loc", [FLAGS.hidden_layers_dim])
            scale = tf.nn.softplus(tf.get_variable("scale", [FLAGS.hidden_layers_dim]))
            tf.get_variable(Normal(loc=loc, scale=scale), name="qb_"+str(i))

    with tf.variable_scope("qW_" + str(FLAGS.hidden_layers_dim)):
        loc = tf.get_variable("loc", [FLAGS.hidden_layers_dim, 1])
        scale = tf.nn.softplus(tf.get_variable("scale", [FLAGS.hidden_layers_dim, 1]))
        tf.get_variable(Normal(loc=loc, scale=scale), name="qW_"+str(FLAGS.hidden_layers_dim))
    with tf.variable_scope("qb_" + str(FLAGS.hidden_layers_dim):
        loc = tf.get_variable("loc", [1])
        scale = tf.nn.softplus(tf.get_variable("scale", [1]))
        tf.get_variable(Normal(loc=loc, scale=scale), name="gb_"+str(FLAGS.hidden_layers_dim))

inference = ed.ReparameterizationKLqp({W_0: qW_0, b_0: qb_0,
                       W_1: qW_1, b_1: qb_1,
                       W_2: qW_2, b_2: qb_2}, data={X: x_train, y: y_train})

# log to tensorboard
inference.run(logdir='log')

if __name__ == "__main__":
  tf.app.run()