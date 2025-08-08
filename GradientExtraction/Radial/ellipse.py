import numpy as np
from misc import tensor2numpy
from scipy.optimize import least_squares
from tools import timer
from Radial.mikes_method import pointNearCenter


def con_centric(Gr, Gg, Gb, o):
    rows, cols = Gr.shape[0], Gr.shape[1]
    o_init = o
    e_init = np.array([0.5, 0.5])
    a_init = np.array([rows/2])
    # e_init = np.array([0., 0.])

    # ortho_mat = np.array([[0, -1], [1, 0]])
    p = np.stack(list(np.ndindex((rows, cols))))
    # flip xy and inverse y because coordinate systems
    p = np.fliplr(p)
    p[:, 1] = cols - p[:, 1] - 1

    linear_grads = np.concatenate([Gr.reshape(-1, 2), Gg.reshape(-1, 2), Gb.reshape(-1, 2)])
    x0 = np.concatenate([o_init, e_init, a_init])
    print("Starting Conf:", x0)

    def fun(x):
        o, e, a = x[:2], x[2:4], x[4]
        c1_p, c2_p = np.tile(p-(o-a*e), [3,1]), np.tile(p-(o+a*e), [3,1])
        r1, r2 = np.linalg.norm(c1_p, axis=1), np.linalg.norm(c2_p, axis=1)
        f = np.abs(r2*((c1_p * linear_grads).sum(axis=1)) - r1*((c2_p * linear_grads).sum(axis=1)))
        c = np.abs(r1 + r2 - 2*a)
        return f + c

    res = least_squares(fun, x0)
    o, e, a = res.x[:2], res.x[2:4], res.x[4]
    print("Final Conf:", res.x)
    return o, e, a


def non_concentric_center_eccentricity(Gr, Gg, Gb, o):
    rows, cols = Gr.shape[0], Gr.shape[1]
    o_init = o
    e_init = np.array([0.0, 0.0])
    ortho_mat = np.array([[0, -1], [1, 0]])
    p = np.stack(list(np.ndindex((rows, cols))))

    # flip xy and inverse y because coordinate systems
    p = np.fliplr(p)
    p[:, 1] = cols - p[:, 1] - 1
    # rs=np.linalg.norm(p-o_init[None,:],axis=1)

    linear_grads = np.concatenate([Gr.reshape(-1, 2), Gg.reshape(-1, 2), Gb.reshape(-1, 2)])
    linear_grads = linear_grads @ ortho_mat

    scale = np.array([1,1])

    x0 = np.concatenate([o_init, e_init, scale])

    def calc_rs(o, e, scale):
        op = (o[None, :] - p)
        a = (e.T @ e - 1)
        b = 2 * (e[None, :] * op).sum(axis=1)
        c = (op * op).sum(axis=1)

        ret = (-b - np.sqrt(-(4 * a * c) + (b ** 2))) / (2 * a)

        return ret

    def unpack_x(x):
        return x[:2], x[2:4], x[4:6]

    def fun(x):
        o, e, sale = unpack_x(x)
        rs = calc_rs(o, e, scale)
        f = (linear_grads*scale * np.tile((p - o[None, :] - rs[:, None] * e[None, :])*scale, [3, 1])).sum(axis=1)
        return f

    # res=scipy.optimize.minimize(fun,x0,method='L-BFGS-B')
    res = least_squares(fun, x0)

    o, e, scale = unpack_x(res.x)
    rs = calc_rs(o, e, scale)

    return o, e, scale, rs.reshape([rows, cols]), res.cost, res.fun.reshape(3, *[rows, cols])


def foci(Gs):
    Gr, Gg, Gb = Gs[0], Gs[1], Gs[2]
    rows, cols = Gr[0].shape[0], Gr[0].shape[1]

    Gr = np.stack([tensor2numpy(Gr[0]), tensor2numpy(Gr[1])], axis=2)
    Gg = np.stack([tensor2numpy(Gg[0]), tensor2numpy(Gg[1])], axis=2)
    Gb = np.stack([tensor2numpy(Gb[0]), tensor2numpy(Gb[1])], axis=2)
    o = pointNearCenter(Gr, Gg, Gb)
    o, e, scale, r, oe_cost, oe_residuals = non_concentric_center_eccentricity(Gr, Gg, Gb, o)
    # find origin assuming concentric


    return o, e, scale
