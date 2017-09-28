import numpy as np

u = np.array([[1, -3], [4, -1]])
v = np.array([[1, 1], [-1, -1]]) * 0.5
w = np.array([[1, -1], [1, -1]]) * 0.5

f = np.array([[-2, 6, 3], [13, 7, 5], [7, 1, 8], [-3, 3, 4]])
o1 = np.array([[0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]]) / 3
o2 = np.array([[1, 1, 1], [1, 0, 1], [-1, -1, -1], [0, -1, 0]]) / 3
o3 = np.array([[1, 0, -1], [1, 0, -1], [0, 0, 0], [0, 0, 0]]) * 0.5
o4 = np.array([[0, 0, 0], [0, 0, 0], [1, 0, -1], [1, 0, -1]]) * 0.5

print(np.sum(np.multiply(f, o1)))
print(np.sum(np.multiply(f, o2)))
print(np.sum(np.multiply(f, o3)))
print(np.sum(np.multiply(f, o4)))

print(16 * o1 + 2 * o2 + 1.5 * o3 - 4 * o4)

# print(np.sum(np.multiply(o1, o2)))
# print(np.sum(np.multiply(o1, o3)))
# print(np.sum(np.multiply(o1, o4)))
# print(np.sum(np.multiply(o2, o3)))
# print(np.sum(np.multiply(o3, o4)))