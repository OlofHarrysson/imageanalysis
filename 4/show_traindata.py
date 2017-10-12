import scipy.io
import numpy as np
import sys
import matplotlib.pyplot as plt

################################### START ###################################

mat = scipy.io.loadmat('ocrsegments.mat')
x = mat['S'].T

for xi in x: # 100 images
  print(xi[0].shape)
  img = xi[0]
  fig = plt.figure()
  imgplot = plt.imshow(img, cmap='gray')
  plt.show()

