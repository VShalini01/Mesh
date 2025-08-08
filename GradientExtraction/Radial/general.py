import numpy as np
from misc import Matrix
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from misc import  get_circle


Orth = np.array([
    [0, -1],
    [1, 0]
])

I = np.array([
    [1, 0],
    [0, 1]
])


def RMatrix(theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return R


def dot(A, B):
    return np.einsum('ij, ij->i', A, B)


def cross(A, B):
    return dot(A @ Orth, B)


def crop_img(img, c=10):
    return img[c:-c, c:-c, :]


def coord_correct(img):
    """
    +Y is downward in image. Correcting this
    :param img: +Y is upward
    :return:
    """
    return img[::-1]


def conv2d(a, f, mode='same'):
    if mode == 'same':
        p0, p1 = int(f.shape[0] / 2), int(f.shape[1] / 2)
        a = np.pad(a, [(p0,p0), (p1,p1)], mode='edge')
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape=s, strides=a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)


def grad(color_img, normalized = True):
    """
    :param color_img: 1 channel Image is numpy array. Y positive is downwards
    :param normalized: Normalize the gradient.
    :return: gradient along x and y axis. y positive is down.
    """
    assert len(color_img.shape) == 2
    gy, gx =  np.gradient(color_img) #sobel_op(color_img) #
    g = np.stack([gx.flatten(), gy.flatten()], axis=1)
    mag = np.linalg.norm(g, axis=1) if normalized else np.ones(len(g))
    mask = (mag < 1e-8).astype(int)
    mag = mag + mask # convert small values to 1
    return g / mag[:, None]


def points(rows_cols):
    rows, cols = rows_cols[0], rows_cols[1]
    p = np.array(list(np.ndindex(rows, cols)))
    return np.fliplr(p)


def radius(p, o, e):
    op = (o[None, :] - p)
    a = (e.T @ e - 1)
    b = 2 * (e[None, :] * op).sum(axis=1)
    c = (op * op).sum(axis=1)
    dis = np.clip(-(4 * a * c) + (b ** 2), a_min=0, a_max=np.inf)
    return (-b - np.sqrt(dis)) / (2 * a)


def img_points_grad(img, crop_edges=None):
    """
    :param img:
    :return: img, grad will be cropped, grad and points are flatten.
    They progress as (0,0) ... (max_x, 0),(0,1) ... (max_x, 1), ... ... ,(0, max_y) ... (max_x, max_y)
    Where max_x, max_y is the number of cols and rows, respectively in the cropped
    """
    if img is None:
        exit(0)
    assert len(img.shape) == 3
    img = crop_img(img, c=crop_edges) if crop_edges is not None else img

    Gs = [
        grad(img[:,:,i]) for i in range(3)
    ]

    mask = img[:,:,3].flatten() if img.shape[2] == 4 else None
    P = points(img.shape[:2])
    mask3 = np.concatenate([mask, mask, mask]) if mask is not None else np.ones(3 * len(P))
    rows, cols = img.shape[0], img.shape[1]
    cols_rows = np.array([cols, rows])
    # concat color and point gradient for three channels
    Gs = np.concatenate([G for G in Gs])
    P = np.tile(P, [3, 1])

    return img, P, Gs, mask3, cols_rows


def normalize(F):
    mag = np.linalg.norm(F, axis=1)
    mask = (mag < 1e-10).astype(int)
    mag = mag + mask  # convert small values to 1
    return F / mag[:, None]


def fit(img):
    # Image is defined in space J
    img, P, Gs, mask, cols_rows = img_points_grad(img, crop_edges=None)

    def theta(P, o, f, r):
        of = o - f
        fP = f - P
        a = of.T @ of - r**2
        b = 2* (of * fP).sum(axis=1)
        c = dot(fP, fP)
        dis = np.clip(-(4 * a * c) + (b ** 2), a_min=0, a_max=np.inf)
        return (-b - np.sqrt(dis)) / (2 * a)

    def func(x):
        fHat, oHat, rHat = x[:2], x[2:4], x[4]

        PHat = P # Space I and J are identical
        t = theta(P, oHat, fHat, rHat)[:, None]
        cPHat = fHat*(1-t) + oHat*t
        GHat = PHat - cPHat
        res = cross(Gs, GHat)
        print("Residue: Sum:{}, Focus:{}, Origin:{}, Radius:{}".format(np.sum(np.abs(res)), fHat, oHat, rHat))
        return res

    def pack(f, o, r): return np.concatenate([f, o, [r]])

    # Following parameters are defined in non-rotated iso-metric space I. Denoted by hat
    fHat, oHat, rHat = cols_rows*0.5, cols_rows*0.5, cols_rows[0]*0.5 # Initialized by arbitrary value
    residue = least_squares(fun=func, x0=pack(fHat, oHat, rHat))
    return img, residue.x[:2], residue.x[2:4], residue.x[4]


def print_trained_params(f, o, e, s, t, rad):
    print("-------------------------------------"
          "\n\tCenter \t\t :[{:.3f}, {:.3f}] "
          "\n\tFocal  \t\t :[{:.3f}, {:.3f}] "
          "\n\tEccentricity :[{:.3f}, {:.3f}] "
          "\n\tRadius :[{}] "
          "\n\tScale    :{:.3f} "
          "\n\tRotation :{:.3f} deg\n"
          "-------------------------------------".
          format(f[0], f[1], o[0], o[1], e[0], e[1], rad, s, np.rad2deg(t)))


def draw_gradient(N, G, P, ax, mask):
    l = 0.5
    for i, p in enumerate(P[:N]):
        if mask[i] < 1: continue
        mag = np.linalg.norm(G[i])
        if mag < 1e-10: continue
        q = p + G[i]/mag*l
        ax.plot([p[0], q[0]], [p[1], q[1]], 'b-')
    ax.set_aspect('equal')


def ecc_direc(f, e, cols_rows):
    l = cols_rows[0]*0.4
    return np.array([[f[0], f[1]], [f[0]+l*e[0], f[1]+l*e[1]]])


def draw_reconstruction(img, f, o, r):
    from misc import get_circle
    plt.subplots()
    plt.imshow(img)
    plt.plot(f[0], f[1], 'o')
    plt.plot(o[0], o[1], '+')
    circ = get_circle(o, r, 1, 0)
    plt.plot(circ[:,0], circ[:,1], '--')
    plt.show()


def main(file):
    from misc import read_img
    img = read_img(file + '.png')
    if img is None:
        print ("ERROR: Image not found.!!")
        exit(1)
    image, fHat, oHat, rHat = fit(img)
    print_trained_params(fHat, oHat, np.zeros(2), 0, 0, rHat)
    draw_reconstruction(img, fHat, oHat, rHat)


if __name__ == '__main__':
    main('Con200')





