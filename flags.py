from absl import flags
import os

flags.DEFINE_string("data_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
        "bayesian_neural_network/data"),
    help="Directory where data is stored (if using real data).") # ?

flags.DEFINE_string("model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
        "bayesian_neural_network/"),
    help="Directory to put the model's fit.") # ?

flags.DEFINE_float("learning_rate",
    default=0.01,
    help="Initial learning rate.")

flags.DEFINE_integer("max_epochs",
    default=6000,
    help="Number of training epochs to run.")

flags.DEFINE_integer("num_hidden_layers",
    default=2,
    help="Number of hidden layers")

flags.DEFINE_integer("num_neurons_per_layer",
    default=50,
    help="Number of neurons per hidden layer")

flags.DEFINE_list("layer_sizes",
    default=["128", "128", "128"],
    help="Comma-separated list denoting hidden units per layer.")

flags.DEFINE_string("activation_function",
    default="relu",
    help="Activation function for all hidden layers.")

flags.DEFINE_integer("batch_size",
    default=44,
    help="Batch size. Must divide evenly into dataset sizes.")

flags.DEFINE_integer("num_monte_carlo",
    default=100,
    help="Network draws to compute predictive probabilities.")


flags.DEFINE_integer("num_epochs",
    default=10000,
    help="Number of epochs to run the training for.")

flags.DEFINE_string("hyperparams_dir",
    default="hyperparams.json",
    help="Directory to the json for the hyperparameters")

flags.DEFINE_integer("num_principal_components",
    default=200,
    help="Number of principal components to reduce the dataset into.")