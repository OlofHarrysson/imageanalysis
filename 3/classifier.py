import scipy.io
import numpy as np
import sys
import numpy.linalg as LA
import matplotlib.pyplot as plt

def nearest_neighbour(train_x, train_y, test_x, test_y, n_img_to_plot=0):
  nbr_correct = 0

  for test_x_i, test_y_i in zip(test_x, test_y):
    if n_img_to_plot > 0:
      plot(classify_nn(train_x, train_y, test_x_i), test_y_i, test_x_i) # Plots image and classification
      n_img_to_plot -= 1

    nbr_correct += 1 if classify_nn(train_x, train_y, test_x_i) == test_y_i else 0

  return nbr_correct / test_x.shape[0]


def classify_nn(train_x, train_y, test_x):
  norms = LA.norm(train_x - test_x, ord=2, axis=1)
  return train_y[np.argmin(norms)]


def plot(prediction, ground_truth, img):
  img = np.reshape(img, (19, 19), order='F')
  fig = plt.figure()
  nbr_to_face = lambda x: 'face' if x==[1] else 'not a face'

  if prediction == ground_truth:
    plt.title('Correctly classified as {}'.format(nbr_to_face(prediction)))
  else:
    plt.title('Incorrectly classified as {}'.format(nbr_to_face(prediction)))

  imgplot = plt.imshow(img, cmap='gray')
  plt.show()

################################### START ###################################

mat = scipy.io.loadmat('FaceNonFace.mat')
x = mat['X'].T.astype(np.int16)
y = mat['Y'].T

accuracies = []
n_loops = 1
for i in range(n_loops):
  msk = np.random.rand(len(x)) < 0.8

  train_x = x[msk]
  train_y = y[msk]

  test_x = x[~msk]
  test_y = y[~msk]

  accuracies.append(nearest_neighbour(train_x, train_y, test_x, test_y, n_img_to_plot=3))

print("Mean accuracy of {} loops: {}".format(n_loops, np.mean(accuracies)))

