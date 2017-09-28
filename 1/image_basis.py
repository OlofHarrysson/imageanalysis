import scipy.io
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

mat = scipy.io.loadmat('assignment1bases.mat')
stacks = mat['stacks'].flatten()
bases = mat['bases'].flatten()

def calc_mean(stack, base, n_img_to_plot=0):
    mean_err = []
    for img in stack: # For every image

        # Does the projection. Return projected image and err_vector
        up_img, err_norm = project(img, base)


        if n_img_to_plot > 0:
            plot(img, up_img) # Plots image and projected image
            n_img_to_plot -= 1

        mean_err.append(err_norm)

    return np.mean(mean_err) # Returns err_mean for the current stack+base


def project(u_img, base):
    img_vector = np.hstack(u_img)
    up_img = np.zeros_like(img_vector)
    for b_ele in base:
        b_ele = np.hstack(b_ele)
        up_img = up_img + np.dot(img_vector, b_ele) * b_ele

    up_img = np.reshape(up_img, u_img.shape)

    err_norm = norm(u_img - up_img)
    return up_img, err_norm

def norm(img):
    img_vector = np.hstack(img)
    img_vector = np.power(img_vector, 2)
    return np.sqrt(np.sum(img_vector))


def plot(img, up_img):
    fig = plt.figure()
    a = fig.add_subplot(1,2,1)
    imgplot = plt.imshow(img, cmap='gray')
    a = fig.add_subplot(1,2,2)
    imgplot = plt.imshow(up_img, cmap='gray')
    plt.show()


def plot_base(base):
    gs = gridspec.GridSpec(2, 2)

    ax1 = plt.subplot(gs[0])
    ax1.imshow(base[0], cmap='gray')

    ax2 = plt.subplot(gs[1])
    ax2.imshow(base[1], cmap='gray')

    ax3 = plt.subplot(gs[2])
    ax3.imshow(base[2], cmap='gray')

    ax4 = plt.subplot(gs[3])
    ax4.imshow(base[3], cmap='gray')
    plt.show()


for stack in stacks: # For every stack
    stack = np.rollaxis(stack, 2) # Change axises
    np.random.shuffle(stack) # Change image order to not always plot the first ones

    for base in bases: # For every base
        base = np.rollaxis(base, 2) # Change axises

        # plot_base(base)

        mean = calc_mean(stack, base, n_img_to_plot=0)
        print("Mean: {:f}".format(mean))
