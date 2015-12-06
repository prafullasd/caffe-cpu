"""
Generate data used in the HDF5DataLayer and GradientBasedSolver tests.
"""
import os
import numpy as np
import h5py

script_dir = os.path.dirname(os.path.abspath(__file__))

# Generate HDF5DataLayer sample_data.h5

num_cols = 4
num_rows = 100
height = 1
width = 1
total_size = num_cols * num_rows * height * width

data = np.random.randn(num_rows, num_cols, height, width)
data = data.reshape(num_rows, num_cols, height, width)
data = data.astype('float32')
label = np.zeros((num_rows, 1, 1, 1))
for i in range(num_rows):
    label[i][0] = np.argmax(data[i]);

print data
print label

with h5py.File(script_dir + '/sample_data.h5', 'w') as f:
    f['data'] = data
    f['label'] = label

