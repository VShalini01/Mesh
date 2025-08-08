import matplotlib.pyplot as plt
import numpy as np
from misc import tensor2numpy, sanitize


def image_direction(image, theta, label=None):
    img = sanitize(image)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    if label is not None:
        ax.set_title(label)
    ax.imshow(img)
    start = np.array([img.shape[1]/2, img.shape[0]/2])
    length = np.min(img.shape[0:2])/3

    degree_sign = u"\u00b0"
    print("\nDirection in degrees :{:.2f}{}".format(np.rad2deg(theta), degree_sign))
    ax.quiver(start[0], start[1], length*np.cos(theta), length*np.sin(theta), units='xy', scale=1, color='red')
    # end = 3*np.array([np.cos(direction), np.sin(direction)])
    # ax.arrow(start[0], start[1], length*np.cos(direction), length*np.sin(direction))
    plt.show()


def vector_field(img, mask, Fx, Fy, label=None):
    assert Fx.shape == Fy.shape
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    if label is not None:
        ax.set_title(label)

    skip = int(min(Fx.shape)/30)
    length = skip/2
    for i,j in np.ndindex(Fx.shape):
        if i % skip == 0 and j % skip == 0 and 0 < i < Fx.shape[0]-1 and 0 < j < Fx.shape[1]-1 and \
                not (Fx[i, j] == 0 or Fy[i, j] == 0):
            mag = np.sqrt(Fx[i,j]**2 + Fy[i, j]**2)
            gx, gy = Fx[i,j]/mag, Fy[i, j]/mag
            x,y = j+0.5, i +0.5
            color = 'red'
            ax.plot([x, x+length*gx], [y, y + -length*gy], '-', color=color)
            # ax.quiver(x, y, length*gx, length*gy, units='xy', scale=1, color='red')
    if len(img.shape) == 3 and img.shape[0] == 3:
        ax.imshow(sanitize(img))
    else:
        ax.imshow(img, cmap='gray', alpha=mask)

    plt.show()


def draw_vector_field(grad_field, point, ax, cols_rows, color='blue', label=''):
    """
    :param grad_field: Shape rows*cols x 2
    :param point: points coordinates
    :return: draws the gradient filed. y+ is up
    """
    assert grad_field.shape == point.shape
    print("Drawing Gradient Field : {}".format(label))
    d = 0.5
    D = int(np.min(cols_rows)/50) if np.min(cols_rows) > 128 else 1
    lx, ly = D * d, D * d

    mag = np.linalg.norm(grad_field, axis=1).reshape(-1, 1)
    mag[mag <1e-5] = 1
    grad_field = grad_field / mag
    ax.set_aspect('equal')
    ax.set_title(label)

    cols = int(cols_rows[0])
    def index(i):
        return  int(i/cols), i % cols

    for i, a, b in zip(range(point.shape[0]), point, grad_field):
        k,l = index(i)
        if k % D == 0 and l % D ==0:
            ax.plot([a[0], a[0] + lx * b[0]], [a[1], a[1] + ly * b[1]], '-', color=color, markersize=1)
