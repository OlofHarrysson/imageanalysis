import numpy as np

p1 = [[3, 2, 1, 0], [2, 2, 2, 0], [2, 1, 2, 1]]

p2 = [[1, 2, 2, 3], [1, 1, 0, 2], [3, 1, 2, 0]]

F = np.array([[-4, 2, -6], [3, 0, 7], [-6, 9, 1]])

a = np.array([[1,2], [3,2], [0,3]])
b = np.array([[1,1], [5,1], [-1, -3]])


correponding_points = []
for ai in a:
  x1 = np.append(ai, 1)

  for bi in b:
    x2 = np.append(bi, 1)[np.newaxis] # Add axis to transpose

    if x1.dot(F).dot(x2.T) == 0:
      correponding_points.append((ai, bi))


for p1, p2 in correponding_points:
  print("point {} corresponds to {}".format(p1, p2))


