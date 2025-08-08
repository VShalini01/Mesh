import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from tools import timer
from ellipse_to_svg import create_svg
from misc import Matrix


TO_SVG=True
orthogonal = np.array([
    [0, -1],
    [1, 0]
])

identity = np.array([
    [1, 0],
    [0, 1]
])

COLOR = ['red', 'green', 'blue']


def read(path):
    import cv2
    """ Reads file and returns the image in RGBA"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Error: No image at {}".format(path))
        return
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def grad(color_img):
    """
    :param color_img: 1 channel Image is numpy array. Y positive is downwards
    :return: gradient along x and y axis. y positive is down.
    """
    assert len(color_img.shape) == 2
    cols, rows = color_img.shape[1], color_img.shape[0]
    gy, gx = np.gradient(color_img)
    return np.stack([gx.flatten()/cols, gy.flatten()/rows], axis=1)


def color_grads(img):
    """
    :param img: 3 channel Image is numpy array. Y positive is downwards
    :return: X and Y gradient for each color channel. Y+ is downwards
    """
    assert len(img.shape) == 3 and img.shape[-1] == 3
    return grad(img[:, :, 0]), grad(img[:, :, 1]), grad(img[:, :, 2])


def radius(p, o, e):
    op = (o[None, :] - p)
    a = (e.T @ e - 1)
    b = 2 * (e[None, :] * op).sum(axis=1)
    c = (op * op).sum(axis=1)
    dis = np.clip(-(4 * a * c) + (b ** 2), a_min=0, a_max=np.inf)
    return (-b - np.sqrt(dis)) / (2 * a)


def points(rows_cols):
    rows, cols = rows_cols[0], rows_cols[1]
    p = np.array(list(np.ndindex(rows, cols)))
    return np.fliplr(p)


def star_grad(g):
    return g @ orthogonal


def pack(center, eccentricity, scale, rotation):
    return np.array([center[0], center[1], eccentricity[0], eccentricity[1], scale, rotation])


def draw_vector_field(grad_field, point, ax, cols_rows, color='blue', label=''):
    """
    :param grad_field: Shape rows*cols x 2
    :param point: points coordinates
    :return: draws the gradient filed. y+ is up
    """
    assert grad_field.shape == point.shape

    d = 0.5
    D = int(np.min(cols_rows)/10) if np.min(cols_rows) > 128 else 1
    lx, ly = d/cols_rows[0], d/cols_rows[1]
    mag = np.linalg.norm(grad_field, axis=1).reshape(-1, 1)
    mag[mag == 0] = 1
    grad_field = grad_field / mag
    ax.set_aspect('equal')
    ax.set_title(label)

    for i, a, b in zip(range(point.shape[0]), point, grad_field):
        if i % D ==0:
            ax.plot([a[0], a[0] + lx * b[0]], [a[1], a[1] + ly * b[1]], '-', color=color, markersize=1)


def draw_scalar(s, rows, cols, ax):
    l_v, h_v = np.min(s), np.max(s)
    s = s.reshape(rows, cols)
    ax.set_aspect('equal')
    ax.pcolormesh(np.arange(0, cols, 1), np.arange(0, rows, 1), s, cmap='gray', vmin=l_v, vmax=h_v)


def p2c_exRS(p, c, e, r, s):
    R_1 = Matrix(1, -r)
    q = p @ R_1 @ Matrix(s, 0)
    c = c @ R_1 @ Matrix(s, 0)
    e = e @ R_1 @ Matrix(s, 0)
    rad = radius(q, c, e)
    re = rad[:, None] * e[None, :]
    ce = c[None, :] + re
    ce2q = (ce - q) @ Matrix(s, r)
    return ce2q


def crop_img(img, c=10):
    return img[c:-c, c:-c, :]

@timer
def get_radial_grad_param(img):
    if img is None:
        exit(0)
    assert len(img.shape) == 3 and img.shape[-1] == 3
    img = crop_img(img)

    rows, cols = img.shape[0], img.shape[1]
    cols_rows = np.array([cols, rows])
    print("Rows:{}, Cols:{}".format(rows, cols))

    # Constants
    gs = color_grads(img)
    gs_ortho = np.concatenate([star_grad(g) for g in gs])
    print("Gradient shape:{}. Concatenated Gradient:{}".format(gs[0].shape, gs_ortho.shape))

    p = points([rows, cols]) / cols_rows
    origin = np.ones(2) * 0.5
    normalized_point = p - origin[None, :]

    # Independent Variables : Intialized
    c = origin # np.array([0.2276, 0.3310])
    ecc = np.array([0,0])
    s = 1
    t = 0

    def unpack(x):
        """
        :param x: array of independent variables to be trained
        :return: center, eccentricity, scale, rotational theta
        """
        assert x.shape == (6,)
        return x[:2], x[2:4], x[4], x[5]

    def TxGp(x):
        c, ecc, s, theta = unpack(x)
        normalized_center = c - origin
        Gp = p2c_exRS(normalized_point, normalized_center, ecc, theta, s)
        return Gp

    def optimize(x):
        constructed_grad = TxGp(x)
        constructed_grad = np.tile(constructed_grad, [3,1])
        return np.einsum('ij,ij->i', gs_ortho, constructed_grad)

    residue = least_squares(optimize, x0=pack(c, ecc, s, t), xtol=1e-5, ftol=1e-5)

    res = optimize(residue.x)
    print("Residue: Sum {:.4f}, max: {:.3f}, min: {:.3f}".format(np.sum(np.abs(res)), np.max(res), np.min(res)))
    return unpack(residue.x), gs, normalized_point, cols_rows, TxGp


def print_trained_params(c, ecc, s, t):
    print("-------------------------------------"
          "\n\tCenter \t\t :[{:.3f}, {:.3f}] \n\tEccentricity :[{:.3f}, {:.3f}] \n\tScale    :{:.3f} \n\tRotation :{:.3f} deg\n"
          "-------------------------------------".
          format(c[0], c[1], ecc[0], ecc[1], s, np.rad2deg(t)))


def reconstruct(c, ecc, s, t, gs, normalized_point, cols_rows, img, TxGp, file_name='None'):
    fig, ax = plt.subplots(1,2)
    draw_vector_field(gs[0], normalized_point, ax[0], cols_rows, color='red')
    x = pack(c, ecc, s, t)
    label = ("File:" + file_name ) if file_name is not None else None
    draw_vector_field(TxGp(x), normalized_point, ax[0], cols_rows, color='blue', label=label)

    ax[1].imshow(img)
    c_rec = c * cols_rows
    ax[1].plot(c_rec[0], c_rec[1], 'o', color='blue')
    ecc_ray = c_rec + ecc * cols_rows * 0.5
    ax[1].plot([c_rec[0], ecc_ray[0]], [c_rec[1], ecc_ray[1]], '-', color='blue')

    N = 100
    l = np.arange(0, 2*np.pi, 2*np.pi/N)
    cp = np.stack([np.cos(l), np.sin(l)]).T
    cp = cp @ np.array([[1,0],[0,s]]) @ Matrix(1, t)
    cp = cp * cols_rows * 0.5 + ecc_ray
    ax[1].plot(cp[:,0], cp[:,1], '--')

    plt.show()


def show(c, ecc, s, t, gs, normalized_point, cols_rows, TxGp):
    x = pack(c, ecc, s, t)

    fig, ax = plt.subplots(1, 2)
    draw_vector_field(gs[0], normalized_point, ax[0], cols_rows, color='red')
    draw_vector_field(TxGp(x), normalized_point, ax[1], cols_rows, color='blue', label="Transformed Constructed Gradient")
    plt.show()


def read_img(name = None):
    import os
    from hyperparam import IMAGE_STOCK
    # IMAGE_STOCK = "/Users/souchakr/GIT/GradientExtraction/Image_stock/"
    img = read(os.path.join(IMAGE_STOCK, 'Radial', 'General', 'ecc_r45.png' if name is None else name))
    return img


def run_tests(file_names):
    for file in file_names:
        img = read_img(file+'.png')
        if img is None:
            continue

        x, gs, normalized_point, cols_rows, TxGp = get_radial_grad_param(img)
        print_trained_params(*x)

        if not TO_SVG:
            reconstruct(*x, gs, normalized_point, cols_rows, img, TxGp, file_name=file)
        else:
            create_svg(img, *x)


if __name__ == '__main__':
    files = [
        'ecc256', 'ecc256_a', 'ecc256_scaled', 'ecc256_scaled_a', 'ecc512', 'ecc512_a'
    ]
    run_tests(['ecc_scale_a'])
