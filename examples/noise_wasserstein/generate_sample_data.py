"""
Generate data used in the HDF5DataLayer and GradientBasedSolver tests.
"""
import os
import numpy as np
import h5py
import random
import pylab as pl

script_dir = os.path.dirname(os.path.abspath(__file__))

# Generate HDF5DataLayer sample_data.h5
d = 7
sigma = 0.25
noise = 0.9

num_cols = 2
height = 1
width = 1
train_rows = 2000
test_rows = 500

def getNeighbors(l):
  neighbors = []
  if (l-d >= 0): neighbors.append(l-d)
  if (l-1 >= 0): neighbors.append(l-1)
  if (l+1 < d*d): neighbors.append(l+1)
  if (l+d < d*d) : neighbors.append(l+d)
  return neighbors

# training data
total_size = num_cols * train_rows * height * width

data = np.zeros((train_rows, num_cols, height, width))
data = data.reshape(train_rows, num_cols, height, width)
data = data.astype('float32')

label = np.zeros((train_rows, 1, 1, 1))

# generate data
for i in range(train_rows):
  x = random.randint(0, d-1)
  y = random.randint(0, d-1)
  data[i][0] = random.gauss(x, sigma)
  data[i][1] = random.gauss(y, sigma)
  oriLabel = x * d + y
  if (random.randint(0,100) < noise * 100):
    neighbors = getNeighbors(oriLabel)
    ind = random.randint(0, len(neighbors) - 1)
    label[i] = neighbors[ind]
  else :
    label[i] = oriLabel




print data
print label

#plot training points
if (False):
  pl.clf()
  pl.scatter(data[:, 0], data[:, 1], c=label, s=50)
  pl.title('sample_data')
  pl.axis('tight')
  pl.show()


with h5py.File(script_dir + '/sample_train_data.h5', 'w') as f:
    f['data'] = data
    f['label'] = label

with open(script_dir + '/sample_train_data_list.txt', 'w') as f:
    f.write('examples/noise_wasserstein/sample_train_data.h5\n')


# test data

total_size = num_cols * test_rows * height * width

data = np.zeros((test_rows, num_cols, height, width))
data = data.reshape(test_rows, num_cols, height, width)
data = data.astype('float32')

label = np.zeros((test_rows, 1, 1, 1))

# generate data
for i in range(test_rows):
  x = random.randint(0, d-1)
  y = random.randint(0, d-1)
  data[i][0] = random.gauss(x, sigma)
  data[i][1] = random.gauss(y, sigma)
  label[i] = x * d + y

print data
print label

with h5py.File(script_dir + '/sample_test_data.h5', 'w') as f:
  f['data'] = data
  f['label'] = label

with open(script_dir + '/sample_test_data_list.txt', 'w') as f:
  f.write('examples/noise_wasserstein/sample_test_data.h5\n')