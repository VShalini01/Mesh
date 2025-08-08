import numpy as np
import cv2

def read(path):
    """ Reads file and returns the image in RGBA"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Error: No image at {}".format(path))
        return None
    if len(img.shape) == 3 and img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    elif len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def crop(img):
    return max(1, int(np.max(img.shape) / 16))


def grad(color_img, normalize=True):
    """
    :param color_img: 1 channel Image is numpy array. Y positive is downwards
    :return: gradient along x and y axis. y positive is down.
    """
    assert len(color_img.shape) == 2
    cols, rows = color_img.shape[1], color_img.shape[0]
    gy, gx = np.gradient(color_img)
    if normalize:
        mag = np.sqrt(gy ** 2 + gx ** 2)
        mask = (mag > 1e-6).astype(int)
        mag[mag<=0] = 1.0
        gx, gy = mask * gx / mag, mask * gy / mag
    return np.stack([gx.flatten()/cols, gy.flatten()/rows], axis=1)


def color_grads(img):
    """
    :param img: 3 channel Image is numpy array. Y positive is downwards
    :return: X and Y gradient for each color channel. Y+ is downwards
    """
    assert len(img.shape) == 3 and img.shape[-1] == 3
    return grad(img[:, :, 0]), grad(img[:, :, 1]), grad(img[:, :, 2])


def read_img(folder, name, crop_edges):
    import os
    from GradientExtraction.Radial.hyperparam import IMAGE_STOCK
    print("Path: ", os.path.join(IMAGE_STOCK, 'Radial', folder, name))
    img = read(os.path.join(IMAGE_STOCK, 'Radial', folder, name))
    img = img[:,:, :3]
    if img is None:
        print ("ERROR: Image not found.!!")
        exit(1)
    from GradientExtraction.Radial.misc import sanitize, smoothen, tensor_img
    img = sanitize(smoothen(tensor_img(img)))
    if crop_edges:
        c = crop(img)
        img = img[c:-c, c:-c, :]
    return img


def radius(p, o, e):
    op = (o[None, :] - p)
    a = (e.T @ e - 1)
    b = 2 * (e[None, :] * op).sum(axis=1)
    c = (op * op).sum(axis=1)
    dis = np.clip(-(4 * a * c) + (b ** 2), a_min=0, a_max=np.inf)
    return (-b - np.sqrt(dis)) / (2 * a)