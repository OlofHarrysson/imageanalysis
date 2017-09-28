import scipy.io
import numpy as np
import sys
import numpy.linalg as LA

def nearest_neighbour(train, test):
  nbr_correct = 0
  for test_x, test_y in test:
    nbr_correct += 1 if classify_nn(test_x, train) == test_y else 0

  return nbr_correct / len(test)

def classify_nn(test_x, train):
  min_error = float("Inf")
  classified_as = None

  for train_x, train_y in train:
    norm = norm1(test_x - train_x)
    if norm < min_error:
      min_error = norm
      classified_as = train_y

  return classified_as

def norm1(img):
  img_vector = np.hstack(img)
  img_vector = np.power(img_vector, 2)
  return np.sqrt(np.sum(img_vector))

################################### START ###################################

mat = scipy.io.loadmat('FaceNonFace.mat')
x = mat['X']
y = mat['Y']

data = np.array(list(zip(x.T, y.T)))
msk = np.random.rand(len(data)) < 0.8
train = data[msk]
test = data[~msk]

accuracy = nearest_neighbour(train, test)
print(accuracy)

