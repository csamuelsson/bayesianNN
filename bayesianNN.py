# Import necessary dependencies
# import matplotlib
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tf.contrib.distributions

def main():
  FLAGS.layer_sizes = [int(units) for units in FLAGS.layer_sizes]
  FLAGS.activation = getattr(tf.nn, FLAGS.activation)
  if tf.gfile.Exists(FLAGS.model_dir):
    tf.logging.warning(
        "Warning: deleting old log directory at {}".format(FLAGS.model_dir))
    tf.gfile.DeleteRecursively(FLAGS.model_dir)
  tf.gfile.MakeDirs(FLAGS.model_dir)

  if FLAGS.fake_data:
    mnist_data = build_fake_data()
  else:
    mnist_data = mnist.read_data_sets(FLAGS.data_dir)

  with tf.Graph().as_default():
    (images, labels, handle,
     training_iterator, heldout_iterator) = build_input_pipeline(
         mnist_data, FLAGS.batch_size, mnist_data.validation.num_examples)

    # Build a Bayesian neural net
    with tf.name_scope("bayesian_neural_net", values=[images]): # values argument??
      neural_net = tf.keras.Sequential()
      for units in FLAGS.layer_sizes:
        layer = tfp.layers.DenseReparameterization(
            units,
            activation=FLAGS.activation)
        neural_net.add(layer)
      neural_net.add(tfp.layers.DenseReparameterization(1)) # one dimensional output
      logits = neural_net(images)
      labels_distribution = tfd.Categorical(logits=logits)

    # Compute the -ELBO as the loss, averaged over the batch size.
    neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(labels))
    kl = sum(neural_net.losses) / mnist_data.train.num_examples
    elbo_loss = neg_log_likelihood + kl

    # Build metrics for evaluation. Predictions are formed from a single forward
    # pass of the probabilistic layers. They are cheap but noisy predictions.
    predictions = tf.argmax(logits, axis=1)
    accuracy, accuracy_update_op = tf.metrics.accuracy(
        labels=labels, predictions=predictions)

    # Extract weight posterior statistics for later visualization.
    names = []
    qmeans = []
    qstds = []
    for i, layer in enumerate(neural_net.layers):
      q = layer.kernel_posterior
      names.append("Layer {}".format(i))
      qmeans.append(q.mean())
      qstds.append(q.stddev())

    with tf.name_scope("train"):
      opt = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)

      train_op = opt.minimize(elbo_loss)
      sess = tf.Session()
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())

      # Run the training loop.
      train_handle = sess.run(training_iterator.string_handle())
      heldout_handle = sess.run(heldout_iterator.string_handle())
      for step in range(FLAGS.max_steps):
        _ = sess.run([train_op, accuracy_update_op],
                     feed_dict={handle: train_handle})

        if step % 100 == 0:
          loss_value, accuracy_value = sess.run(
              [elbo_loss, accuracy], feed_dict={handle: train_handle})
          print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(
              step, loss_value, accuracy_value))

        if (step+1) % FLAGS.viz_steps == 0:
          # Compute log prob of heldout set by averaging draws from the model:
          # p(heldout | train) = int_model p(heldout|model) p(model|train)
          #                   ~= 1/n * sum_{i=1}^n p(heldout | model_i)
          # where model_i is a draw from the posterior p(model|train).
          probs = np.asarray([sess.run((labels_distribution.probs),
                                       feed_dict={handle: heldout_handle})
                              for _ in range(FLAGS.num_monte_carlo)])
          mean_probs = np.mean(probs, axis=0)

          image_vals, label_vals = sess.run((images, labels),
                                            feed_dict={handle: heldout_handle})
          heldout_lp = np.mean(np.log(mean_probs[np.arange(mean_probs.shape[0]),
                                                 label_vals.flatten()]))
          print(" ... Held-out nats: {:.3f}".format(heldout_lp))

          qm_vals, qs_vals = sess.run((qmeans, qstds))

          if HAS_SEABORN:
            plot_weight_posteriors(names, qm_vals, qs_vals,
                                   fname=os.path.join(
                                       FLAGS.model_dir,
                                       "step{:05d}_weights.png".format(step)))

            plot_heldout_prediction(image_vals, probs,
                                    fname=os.path.join(
                                        FLAGS.model_dir,
                                        "step{:05d}_pred.png".format(step)),
                                    title="mean heldout logprob {:.2f}"
.format(heldout_lp))



if __name__ == "__main__":
    tf.app.run()