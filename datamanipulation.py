import pandas as pd
import numpy as np

data = pd.read_csv("variables.csv.txt", sep='\t')

# last column is the response variable
y = np.array(data[data.columns[-1]], dtype='float32')
# rest of the variables (except the first) are the features
x = np.array(data[data.columns[1:-1]], dtype='float32')

# since x_6850 -> x_6908 has variance 0 they don't contribute to our mapping, therefore we remove them here
x = x[0:len(x), 0:6849]

np.savez('drug_data.npz', features=x, labels=y)
with np.load('drug_data.npz') as data:
    x_new = data['features']
    y_new = data['labels']

# Test for NaNs
assert not np.any(np.isnan(x_new))
assert not np.any(np.isnan(y_new))

# Test of equalness in features
for records in zip(x, x_new):
    for elem in zip(records[0], records[1]):
        assert elem[0] == elem[1], "Found two unmatching feature elements"

# Test of equalness in labels
for elem in zip(y, y_new):
    assert elem[0] == elem[1], "Found two unmatching label elements"

print("All tests passed!")