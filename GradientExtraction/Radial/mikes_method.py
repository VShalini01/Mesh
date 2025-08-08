import numpy as np
import torch as th
from misc import grad, smoothen, tensor_img
from hyperparam import VERBOSE
from Draw.draw import sanitize
import matplotlib.pyplot as plt
from misc import tensor2numpy
from scipy.optimize import least_squares
from tools import timer


@timer
def concentric_center(img, Gr, Gg, Gb):
    o_init = np.array([10., 15.])
    ortho_mat = np.array([[0, -1], [1, 0]])

    p = np.stack(np.ndindex(img.shape[1:]))
    # flip xy and inverse y because coordinate systems
    p = np.fliplr(p)
    p[:, 1] = img.shape[2] - p[:, 1] - 1
    p = np.tile(p, [3, 1])

    linear_grads = np.concatenate([Gr.reshape(-1, 2), Gg.reshape(-1, 2), Gb.reshape(-1, 2)])
    linear_grads = linear_grads @ ortho_mat

    def loss_fun(x):
        o = x
        dif = o[None, :] - p
        prod = (dif * linear_grads).sum(axis=1)
        return prod

    def jac(x):
        return linear_grads

    res = least_squares(loss_fun, o_init, jac=jac)

    # normalise residuals by radius
    residuals = res.fun.reshape(3, *Gr.shape[:2])
    radii = np.linalg.norm(np.stack(np.ndindex(img.shape[1:])) - res.x, axis=1).reshape(img.shape[1:])

    residuals = residuals / radii[None, :, :]

    return res.x, res.cost, residuals


def solve(Gr, Gg, Gb, Px, Py):
    '''
    E = Sum_{all points and color} (Gx*(Py-y) - Gy*(Px-x))^2
    dE/dx = 0 = 2 Sum_{all points and color}  Gx*Gy*Py - Gx*Gy*y - Gy*Gy*Px + Gy*Gy*x
    dE/dy = 0 = 2 Sum_{all points and color} -Gx*Gx*Py + Gx*Gx*y + Gx*Gy*Py - Gx*Gy*x
    '''
    A = Gr[:, :, 1] * Gr[:, :, 0] + Gg[:, :, 1] * Gg[:, :, 0] + Gb[:, :, 1] * Gb[:, :, 0]
    B = Gr[:, :, 0] * Gr[:, :, 0] + Gg[:, :, 0] * Gg[:, :, 0] + Gb[:, :, 0] * Gb[:, :, 0]
    C = Gr[:, :, 1] * Gr[:, :, 1] + Gg[:, :, 1] * Gg[:, :, 1] + Gb[:, :, 1] * Gb[:, :, 1]
    a, b, c = np.sum(A), np.sum(B), np.sum(C)
    ay, ax = np.sum(A * Py), np.sum(A * Px)
    by = np.sum(B * Py)
    cx = np.sum(C * Px)

    T = np.array([
        [-a, c],
        [c, -a]
    ])

    R = np.array([
        [cx - ay],
        [by - ax]
    ])

    det = np.linalg.det(T)
    if det == 0:
        print("Error: Cannot solve for radial gradient")
        return None
    X = np.linalg.inv(T) @ R
    x = X[1, 0]
    y = X[0, 0]
    return x, y


@timer
def pointNearCenter(Gr, Gg, Gb):
    row, col = Gr.shape[0], Gr.shape[1]
    Px = np.repeat([np.arange(0, col)], row, axis=0)
    Py = np.repeat([np.arange(0, row)], col, axis=0).T

    x, y = solve(Gr, Gg, Gb, Px, Py)
    arc_point = np.array([x, y])
    return arc_point


@timer
def non_concentric_center_eccentricity(Gr, Gg, Gb, o):
    rows, cols = Gr.shape[0], Gr.shape[1]
    o_init = o
    e_init = np.array([0., 0.])
    ortho_mat = np.array([[0, -1], [1, 0]])
    p = np.stack(list(np.ndindex((rows, cols))))

    # flip xy and inverse y because coordinate systems
    p = np.fliplr(p)
    p[:, 1] = cols - p[:, 1] - 1
    # rs=np.linalg.norm(p-o_init[None,:],axis=1)

    linear_grads = np.concatenate([Gr.reshape(-1, 2), Gg.reshape(-1, 2), Gb.reshape(-1, 2)])
    linear_grads = linear_grads @ ortho_mat

    x0 = np.concatenate([o_init, e_init])

    def calc_rs(o, e):
        op = (o[None, :] - p)
        a = (e.T @ e - 1)
        b = 2 * (e[None, :] * op).sum(axis=1)
        c = (op * op).sum(axis=1)

        ret = (-b - np.sqrt(-(4 * a * c) + (b ** 2))) / (2 * a)

        return ret

    def unpack_x(x):
        return x[:2], x[2:4]

    def fun(x):
        o, e = unpack_x(x)
        rs = calc_rs(o, e)
        f = (linear_grads * np.tile((p - o[None, :] - rs[:, None] * e[None, :]), [3, 1])).sum(axis=1)
        return f

    # res=scipy.optimize.minimize(fun,x0,method='L-BFGS-B')
    res = least_squares(fun, x0)

    o, e = unpack_x(res.x)
    rs = calc_rs(o, e)

    return o, e, rs.reshape([rows, cols]), res.cost, res.fun.reshape(3, *[rows, cols])


def center_eccentricity(Gs):
    Gr, Gg, Gb = Gs[0], Gs[1], Gs[2]
    rows, cols = Gr[0].shape[0], Gr[0].shape[1]

    Gr = np.stack([tensor2numpy(Gr[0]), tensor2numpy(Gr[1])], axis=2)
    Gg = np.stack([tensor2numpy(Gg[0]), tensor2numpy(Gg[1])], axis=2)
    Gb = np.stack([tensor2numpy(Gb[0]), tensor2numpy(Gb[1])], axis=2)

    # find origin assuming concentric
    o = pointNearCenter(Gr, Gg, Gb)
    # now try to find e
    o, e, r, oe_cost, oe_residuals = non_concentric_center_eccentricity(Gr, Gg, Gb, o)

    return np.array([o[0], cols - o[1]]), np.array([e[0], -e[1]]), oe_residuals
