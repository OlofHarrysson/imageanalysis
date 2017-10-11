import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pprint import pprint
import sys

def least_squares(x, y):
  A = np.append(x, np.ones(x.shape), axis=0).T
  p = np.linalg.lstsq(A, y.T)
  k, c = p[0]
  least_square_error = p[1]
  print("Least square error is {}".format(least_square_error))

  return k, c


def total_least_squares(x, y):
  x = x.flatten()
  y = y.flatten()
  n = x.size
  x2 = sum([a**2 for a in np.nditer(x)])
  y2 = sum([a**2 for a in np.nditer(y)])
  xy = np.sum([a*b for a, b in zip(x, y)])
  x_sum = np.sum(x)
  y_sum = np.sum(y)
  n_xy = xy - 1 / n * x_sum * y_sum

  lag_m = [[x2 - 1 / n * x_sum ** 2, n_xy],
           [n_xy, y2 - 1 / n * y_sum ** 2]]

  eig_vals, eig_vecs = np.linalg.eig(lag_m)

  a1 = eig_vecs[0][0]
  b1 = eig_vecs[1][0]

  a2 = eig_vecs[1][0]
  b2 = eig_vecs[1][1]

  c1 = -(1 / n) * (a1 * x_sum + b1 * y_sum)
  c2 = -(1 / n) * (a2 * x_sum + b2 * y_sum)

  y1 = (-c1 - a1 * x) / b1
  y2 = (-c2 - a2 * x) / b2;

  return y1, y2

############# START #############

mat = scipy.io.loadmat('linjepunkter.mat')
x = mat['x']
y = mat['y']

k, c = least_squares(x, y)
y1, y2 = total_least_squares(x, y)

sorted_x = sorted(x[0])
plt.plot(x, y, 'o')
plt.plot(sorted_x, k * sorted_x + c, 'b-', linewidth=2)
plt.plot(x.flatten(), y1, 'g-', linewidth=2)
plt.plot(x.flatten(), y2, 'r-', linewidth=2)
ls_line = mlines.Line2D([], [], color='blue', label='Least Squares')
tls_line = mlines.Line2D([], [], color='green', label='Total Least Squares')
tls_wrong_line = mlines.Line2D([], [], color='red', label='Wrong Total Least Squares')
plt.legend(handles=[ls_line, tls_line, tls_wrong_line])
plt.show()