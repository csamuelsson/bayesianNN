import tensorflow as tf
import numpy as np

import iris_data

# Fetch the data
train_path, test_path = iris_data.maybe_download()

# Build a TextLineDataset object to read the file one line at a time
ds = tf.data.TextLineDataset(train_path).skip(1) # skips the first line

# Build a CSV line parser

# Metadata describing the text columns
COLUMNS = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'label']
FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0]]

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, FIELD_DEFAULTS)

    # Pack the result into a dictionary
    features = dict(zip(COLUMNS, fields))

    # Separate the label from the features
    label = features.pop('label')

    # Return the features and label
    return features, label

# Parse the lines using `map`
ds = ds.map(_parse_line)
print(ds)


# Process data
# batch_size = 50
# iris_data.train_input_fn(features, labels, batch_size)


def train_input_fn(features, labels, batch_size):
    '''An input function for training'''
    # Convert the inputs to a Dataset object
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat and batch the examples
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset
    return dataset