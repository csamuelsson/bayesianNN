from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependencies
from absl import flags
# Local errors with matplotlib, comment out when debugged
# import matplotlib
# matplotlib.use("Agg")
# from matplotlib import figure  # pylint: disable=g-import-not-at-top
# from matplotlib.backends import backend_agg
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.contrib.learn.python.learn.datasets import mnist

# What's happening here? Do we need to care?
# TODO(b/78137893): Integration tests currently fail with seaborn imports.
try:
  import seaborn as sns  # pylint: disable=g-import-not-at-top
  HAS_SEABORN = True
except ImportError:
  HAS_SEABORN = False

tfd = tf.contrib.distributions

IMAGE_SHAPE = [28, 28]

flags.DEFINE_float("learning_rate",
                   default=0.1,
                   help="Initial learning rate.")
flags.DEFINE_integer("max_epochs",
                     default=6000,
                     help="Number of training epochs to run.")
flags.DEFINE_list("layer_sizes",
                  default=["128", "128", "128"],
                  help="Comma-separated list denoting hidden units per layer.")
flags.DEFINE_string("activation",
                    default="relu",
                    help="Activation function for all hidden layers.")
flags.DEFINE_integer("batch_size",
                     default=128,
                     help="Batch size. Must divide evenly into dataset sizes.")
flags.DEFINE_string("data_dir",
                    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                                         "bayesian_neural_network/data"),
                    help="Directory where data is stored (if using real data).") # ?
flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                         "bayesian_neural_network/"),
    help="Directory to put the model's fit.") # ?
flags.DEFINE_integer("viz_epochs",
                     default=400,
                     help="Frequency at which save visualizations.") # ?
flags.DEFINE_integer("num_monte_carlo",
                     default=100,
                     help="Network draws to compute predictive probabilities.")

FLAGS = flags.FLAGS
FLAGS.learning_rate = 0.5


# check hardware
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

# ?
def plot_weight_posteriors(names, qm_vals, qs_vals, fname):
  """Save a PNG plot with histograms of weight means and stddevs.
  Args:
    names: A Python `iterable` of `str` variable names.
    qm_vals: A Python `iterable`, the same length as `names`,
      whose elements are Numpy `array`s, of any shape, containing
      posterior means of weight varibles.
    qs_vals: A Python `iterable`, the same length as `names`,
      whose elements are Numpy `array`s, of any shape, containing
      posterior standard deviations of weight varibles.
    fname: Python `str` filename to save the plot to.
  """
  fig = figure.Figure(figsize=(6, 3))
  canvas = backend_agg.FigureCanvasAgg(fig)

  ax = fig.add_subplot(1, 2, 1)
  for n, qm in zip(names, qm_vals):
    sns.distplot(qm.flatten(), ax=ax, label=n)
  ax.set_title("weight means")
  ax.set_xlim([-1.5, 1.5])
  ax.set_ylim([0, 4.])
  ax.legend()

  ax = fig.add_subplot(1, 2, 2)
  for n, qs in zip(names, qs_vals):
    sns.distplot(qs.flatten(), ax=ax)
  ax.set_title("weight stddevs")
  ax.set_xlim([0, 1.])
  ax.set_ylim([0, 25.])

  fig.tight_layout()
  canvas.print_figure(fname, format="png")
  print("saved {}".format(fname))

# ?
def plot_heldout_prediction(input_vals, probs,
                            fname, n=10, title=""):
  """Save a PNG plot visualizing posterior uncertainty on heldout data.
  Args:
    input_vals: A `float`-like Numpy `array` of shape
      `[num_heldout] + IMAGE_SHAPE`, containing heldout input images.
    probs: A `float`-like Numpy array of shape `[num_monte_carlo,
      num_heldout, num_classes]` containing Monte Carlo samples of
      class probabilities for each heldout sample.
    fname: Python `str` filename to save the plot to.
    n: Python `int` number of datapoints to vizualize.
    title: Python `str` title for the plot.
  """
  fig = figure.Figure(figsize=(9, 3*n))
  canvas = backend_agg.FigureCanvasAgg(fig)
  for i in range(n):
    ax = fig.add_subplot(n, 3, 3*i + 1)
    ax.imshow(input_vals[i, :].reshape(IMAGE_SHAPE), interpolation="None")

    ax = fig.add_subplot(n, 3, 3*i + 2)
    for prob_sample in probs:
      sns.barplot(np.arange(10), prob_sample[i, :], alpha=0.1, ax=ax)
      ax.set_ylim([0, 1])
    ax.set_title("posterior samples")

    ax = fig.add_subplot(n, 3, 3*i + 3)
    sns.barplot(np.arange(10), np.mean(probs[:, i, :], axis=0), ax=ax)
    ax.set_ylim([0, 1])
    ax.set_title("predictive probs")
  fig.suptitle(title)
  fig.tight_layout()

  canvas.print_figure(fname, format="png")
  print("saved {}".format(fname))


# Data pipeline - what's happening here?
def build_input_pipeline(mnist_data, batch_size, heldout_size):
  """Build an Iterator switching between train and heldout data."""
  # Build an iterator over training batches.
  training_dataset = tf.data.Dataset.from_tensor_slices(
      (mnist_data.train.images, np.int32(mnist_data.train.labels)))
  training_batches = training_dataset.repeat().batch(batch_size)
  training_iterator = training_batches.make_one_shot_iterator()

  # Build a iterator over the heldout set with batch_size=heldout_size,
  # i.e., return the entire heldout set as a constant.
  heldout_dataset = tf.data.Dataset.from_tensor_slices(
      (mnist_data.validation.images,
       np.int32(mnist_data.validation.labels)))
  heldout_frozen = (heldout_dataset.take(heldout_size).
                    repeat().batch(heldout_size))
  heldout_iterator = heldout_frozen.make_one_shot_iterator()

  # Combine these into a feedable iterator that can switch between training
  # and validation inputs.
  handle = tf.placeholder(tf.string, shape=[])
  feedable_iterator = tf.data.Iterator.from_string_handle(
      handle, training_batches.output_types, training_batches.output_shapes)
  images, labels = feedable_iterator.get_next()

  return images, labels, handle, training_iterator, heldout_iterator


def main(argv):
  # creates list of the hidden layers where the i:th element represents the i:th hidden layers'
  # dimensions, therefore the length of the list is the number of hidden layers in the model
  FLAGS.layer_sizes = [int(units) for units in FLAGS.layer_sizes]
  # extract the activation function from the hyperopt spec as an attribute from the tf.nn module
  FLAGS.activation = getattr(tf.nn, FLAGS.activation)

  # Not sure what's happening here? but probably nothing too important
  if tf.gfile.Exists(FLAGS.model_dir):
    tf.logging.warning(
        "Warning: deleting old log directory at {}".format(FLAGS.model_dir))
    tf.gfile.DeleteRecursively(FLAGS.model_dir)
  tf.gfile.MakeDirs(FLAGS.model_dir)

  # Load MNIST data
  mnist_data = mnist.read_data_sets(FLAGS.data_dir)

  # define the graph
  with tf.Graph().as_default():
    # what's happening here?
    (images, labels, handle,
     training_iterator, heldout_iterator) = build_input_pipeline(
         mnist_data, FLAGS.batch_size, mnist_data.validation.num_examples)

    # Building the Bayesian Neural Network. 
    # We are here using the Gaussian Reparametrization Trick
    # to compute the stochastic gradients as described in the paper
    with tf.name_scope("bayesian_neural_net", values=[images]): # values arg??
      neural_net = tf.keras.Sequential()
      for units in FLAGS.layer_sizes:
        layer = tfp.layers.DenseReparameterization(
            units,
            activation=FLAGS.activation)
        neural_net.add(layer)
      neural_net.add(tfp.layers.DenseReparameterization(10)) # change to 1
      logits = neural_net(images) # remove
      # predictions = neural_net(images)
      labels_distribution = tfd.Categorical(logits=logits) # remove

      # Extract weight posterior statistics
      names = []
      qmeans = []
      qstds = []
      for i, layer in enumerate(neural_net.layers):
        q = layer.kernel_posterior
        names.append("Layer {}".format(i))
        qmeans.append(q.mean())
        qstds.append(q.stddev())

      # labels_distribution = tfd.MultiVariateNormalDiag(loc=qmeans, scale=qstds)


    # Compute the -ELBO as the loss, averaged over the batch size.
    neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(labels))
    kl = sum(neural_net.losses) / mnist_data.train.num_examples
    elbo_loss = neg_log_likelihood + kl

    # Build metrics for evaluation. Predictions are formed from a single forward
    # pass of the probabilistic layers. They are cheap but noisy predictions.
    predictions = tf.argmax(logits, axis=1) # remove since 1 dimensional output
    accuracy, accuracy_update_op = tf.metrics.accuracy( # chang to mean_squared_error
        labels=labels, predictions=predictions)

    with tf.name_scope("train"):
      # define optimizer - we are using (stochastic) gradient descent
      opt = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)

      # define that we want to minimize the loss (-ELBO)
      train_op = opt.minimize(elbo_loss)
      # start the session
      sess = tf.Session()
      sess.run(tf.global_variables_initializer()) # not sure what this does?
      sess.run(tf.local_variables_initializer()) # note sure what this does?

      # Run the training loop. ?? what is happening from here
      train_handle = sess.run(training_iterator.string_handle())
      heldout_handle = sess.run(heldout_iterator.string_handle())

      # Run the epochs
      for epoch in range(FLAGS.max_epochs):
        _ = sess.run([train_op, accuracy_update_op],
                     feed_dict={handle: train_handle})

        # printing to console every hundred iteration
        if epoch % 100 == 0:
          loss_value, accuracy_value = sess.run(
              [elbo_loss, accuracy], feed_dict={handle: train_handle})
          print("Epoch: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(
              epoch, loss_value, accuracy_value))

        # check if time to save vizualisations
        if (epoch+1) % FLAGS.viz_epochs == 0:
          # Compute log prob of heldout set by averaging draws from the model:
          # p(heldout | train) = int_model p(heldout|model) p(model|train)
          #                   ~= 1/n * sum_{i=1}^n p(heldout | model_i)
          # where model_i is a draw from the posterior p(model|train).
          probs = np.asarray([sess.run((labels_distribution.probs),
                                       feed_dict={handle: heldout_handle}) # will need to change to multivariate normal
                              for _ in range(FLAGS.num_monte_carlo)])
          mean_probs = np.mean(probs, axis=0)

          image_vals, label_vals = sess.run((images, labels),
                                            feed_dict={handle: heldout_handle}) #?
          heldout_lp = np.mean(np.log(mean_probs[np.arange(mean_probs.shape[0]),
                                                 label_vals.flatten()])) #?
          print(" ... Held-out nats: {:.3f}".format(heldout_lp))

          qm_vals, qs_vals = sess.run((qmeans, qstds)) # ?

          if HAS_SEABORN:
            plot_weight_posteriors(names, qm_vals, qs_vals,
                                   fname=os.path.join(
                                       FLAGS.model_dir,
                                       "epoch{:05d}_weights.png".format(epoch)))

            plot_heldout_prediction(image_vals, probs,
                                    fname=os.path.join(
                                        FLAGS.model_dir,
                                        "epoch{:05d}_pred.png".format(epoch)),
                                    title="mean heldout logprob {:.2f}"
                                    .format(heldout_lp))

if __name__ == "__main__":
  tf.app.run()