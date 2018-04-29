from absl import flags
import os

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

flags.DEFINE_string("model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
        "bayesian_neural_network/"),
    help="Directory to put the model's fit.") # ?

flags.DEFINE_integer("viz_epochs",
    default=400,
    help="Frequency at which save visualizations.")

flags.DEFINE_boolean("viz_enabled",
    default=True,
    help="Whether vizualisations should be generated or not")

flags.DEFINE_integer("num_monte_carlo",
    default=100,
    help="Network draws to compute predictive probabilities.")

flags.DEFINE_float("prior_std",
    default=1.0,
    help="Standard deviation for the prior distribution over the weights")

flags.DEFINE_float("prior_pi",
    default=0.5,
    help="Location parameter for the two gaussians that represents the scale mixture prior")

flags.DEFINE_float("prior_sigma_1",
    default=0.75,
    help="Standard deviation for the first gaussian in the scale mixture prior")

flags.DEFINE_float("prior_sigma_2",
    default=0.1,
    help="Standard deviation for the second gaussian in the scale mixture prior")


flags.DEFINE_integer("num_principal_components",
    default=200,
    help="Number of principal components to reduce the dataset towards.")