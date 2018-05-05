import pandas as pd
import numpy as np

data = pd.read_csv("variables.csv.txt", sep='\t')

# last column is the response variable
y = np.array(data[data.columns[-1]], dtype='float32')
# rest of the variables (except the first) are the features
x = np.array(data[data.columns[1:-1]], dtype='float32')

# since x_6850 -> x_6908 has variance 0 they don't contribute to our mapping, therefore we remove them here
x = x[0:len(x), 0:6849]

print("Original data dimensions:", x.shape)

# Remove varuables with 100% correlation with another one
corr_matrix = np.corrcoef(np.transpose(x))

var_to_remove = []
print("Checking for full correlation...")
for i in range(corr_matrix.shape[0]):
    for j in range(corr_matrix.shape[1]):
        if j <= i:
            continue # since symmetry
        elif abs(corr_matrix[i][j]) == 1 & j not in var_to_remove:
            var_to_remove.append(i)

# print(var_to_remove)
x = np.delete(x, var_to_remove, axis=1) # removing variables that are fully correlated with another one
print("New dimension:", x.shape)

# save the data to local file to be loaded later for training
np.savez('drug_data.npz', features=x, labels=y)