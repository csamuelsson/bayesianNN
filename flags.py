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

flags.DEFINE_integer("viz_epochs",
    default=2000,
    help="Frequency at which save visualizations.")

flags.DEFINE_boolean("viz_enabled",
    default=True,
    help="Whether vizualisations should be generated or not")

flags.DEFINE_integer("num_monte_carlo",
    default=100,
    help="Network draws to compute predictive probabilities.")