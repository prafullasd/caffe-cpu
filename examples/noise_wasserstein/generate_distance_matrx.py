"""
Generate distance matrix in hdf5 format.
"""
import os
import numpy as np
import h5py

script_dir = os.path.dirname(os.path.abspath(__file__))

# Generate HDF5DataLayer distance_matrix.h5

d = 7
data = np.arange(d ** 4)
data = data.reshape(d*d, d*d)
data = data.astype('float32')


for i1 in range(d):
  for j1 in range(d):
    for i2 in range(d):
      for j2 in range(d):
        data[i1 * d + j1][i2 * d + j2] = (i1-i2)**2 + (j1-j2)**2

print data

with h5py.File(script_dir + '/sample_dist_mat.h5', 'w') as f:
    f['data'] = data
