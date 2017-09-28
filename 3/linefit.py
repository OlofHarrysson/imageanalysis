import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

mat = scipy.io.loadmat('linjepunkter.mat')
x = mat['x']
y = mat['y']
A = np.append(x, np.ones(x.shape), axis=0).T
p = np.linalg.lstsq(A, y.T)
print(p)
k, c = p[0]
least_square_error = p[1]
print(least_square_error)

sorted_x = sorted(x[0])
plt.plot(x, y, 'o')
plt.plot(sorted_x, k * sorted_x + c, 'b-', linewidth=2)
line = mlines.Line2D([], [], color='blue', label='Least Squares')
plt.legend(handles=[line])
plt.show()

