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

  return k, c, least_square_error[0]


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

def distance_point_line(p1, p2, p3):
  # line goes between p1 and p2.
  p1 = np.asarray(p1)
  p2 = np.asarray(p2)
  p3 = np.asarray(p3)

  return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

############# START #############

mat = scipy.io.loadmat('linjepunkter.mat')
x = mat['x']
y = mat['y']

k, c, least_square_error = least_squares(x, y)
y1, y2 = total_least_squares(x, y)

least_square_point1 = [x.flatten()[0], x.flatten()[0] * k + c]
least_square_point2 = [x.flatten()[-1], x.flatten()[-1] * k + c]


total_lsq_line1_point1 = [x.flatten()[0], y1[0]]
total_lsq_line1_point2 = [x.flatten()[-1], y1[-1]]

least_square_line_sum = [0, 0]
total_lsq_line1_sum = [0, 0]
for x_point, y_point in zip(x.flatten(), y.flatten()):
  least_square_line_sum[1] += np.square(distance_point_line(least_square_point1, least_square_point2, [x_point, y_point]))
  total_lsq_line1_sum[1] += np.square(distance_point_line(total_lsq_line1_point1, total_lsq_line1_point2, [x_point, y_point]))


least_square_line_sum[0] = least_square_error
total_lsq_line1_sum[0] = sum(np.square(np.abs(y1 - y.flatten())))

print("Least square error for the least_square line is {}".format(least_square_line_sum[0]))
print("Least square error for the total_least_square line is {}".format(total_lsq_line1_sum[0]))

print("\nTotal least square error for the least_square line is {}".format(least_square_line_sum[1]))
print("Total least square error for the total_least_square line is {}".format(total_lsq_line1_sum[1]))


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