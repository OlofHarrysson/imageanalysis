import scipy.io
import numpy as np
import sys
import matplotlib.pyplot as plt

################################### START ###################################

mat = scipy.io.loadmat('ocrsegments.mat')
x = mat['S'].T
y = mat['y'][0]
unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))
# print(counts)
# print(y)
sys.exit(1)

for xi in x: # 100 images
  img = xi[0]
  fig = plt.figure()
  imgplot = plt.imshow(img, cmap='gray')
  plt.show()

