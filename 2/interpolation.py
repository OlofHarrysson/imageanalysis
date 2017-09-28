import numpy as np

def interpolate(x):
    total = 0
    for ind, el in enumerate(f):
        total += g(x - ind) * el

    return total

def g(x):
    return 1 - np.abs(x) if np.abs(x) < 1 else 0

f = [1, 4, 6, 8, 7, 5, 3]
inter_pts = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
for point in inter_pts:
    print("{} interpolates to {}".format(point, interpolate(point)))


def deriviate(point1, point2):
    return (point2[1] - point1[1]) / (point2[0] - point1[0])

print("") # spacing
for ind, x2 in enumerate(inter_pts):
    point1 = (ind, f[ind])
    point2 = (x2, interpolate(x2))
    print("Deriviate in {} is {}".format(inter_pts[ind], deriviate(point1, point2)))


w = [1, -1]
print("\nf * w = {}".format(np.convolve(f, w, mode='valid')))

w2 = [0.5, 0, -0.5]
print("\nf * w2 = {}".format(np.convolve(f, w2, mode='valid')))
