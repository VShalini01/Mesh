import numpy as np
from Radial.ellipse3 import Matrix, radius, orthogonal, draw_vector_field #, p2c_exRS
from Radial.ellipse3 import read_img, color_grads, star_grad, points
import matplotlib.pyplot as plt


def p2c_exRS_test(p, c, e, r, s, g, cols_rows):
    N = cols_rows[0]
    assert p.shape[0] == N**2 # rows and cols of size 32 are considered
    lx = p[0:N, 0]
    ly = [p[i * N, 1] for i in range(int(len(p) / N))]

    def index(px, py):
        ind_x, ind_y = None, None
        for ix, v in enumerate(lx[:-1]):
            if v <= px < lx[ix+1]:
                ind_x = ix
                break
        for iy, v in enumerate(ly[:-1]):
            if v <= py < ly[iy+1]:
                ind_y = iy
                break
        ind_x = (N-1) if ind_x is None else ind_x
        ind_y = (N-1) if ind_y is None else ind_y
        return N * ind_y + ind_x


    q = p @ Matrix(1, -r)
    c = c @ Matrix(1, -r)
    e = e @ Matrix(1, -r)
    rad = radius(q, c, e)
    re = rad[:, None] * e[None, :]
    ce = c[None, :] + re

    assert g is not None

    ######################### construction #########################
    construct_draw=True
    if construct_draw:
        fig, ax = plt.subplots(1,2)
        R_1 = Matrix(1, r)
        p_ = p @ R_1
        c_ = c @ R_1
        e_ = e @ R_1
        rad = radius(p_, c_, e_)
        re = rad[:, None] * e[None, :]
        ce = c[None, :] + re
        grad_const = (ce - p) #@ Matrix(1, -r)
        draw_vector_field(g, p, ax[0], cols_rows=np.array([32, 32]), color='red')
        draw_vector_field(grad_const, p, ax[1], cols_rows=np.array([32, 32]), color='blue')
        plt.show()
    ################################################################
    else:
        ######################### Testing #########################
        fig, axs = plt.subplots(1,2)
        ns = [32*(4+16) + 4+16, 32*(4+16) + 16 + 8, 32*(4+16) + 8, 32*(2+16) + 4+16, 32*(2+16) + 16 + 8, 32*(2+16) + 8]
        ax= axs[0]
        ax.plot(c[0], c[1], 'o', color='black')
        draw_vector_field(g, p, ax, cols_rows=np.array([32,32]), color='aqua')

        for n in ns[:3]:
            pn = p[n]
            nt = index(*(pn @ Matrix(1, r)))
            qn = p[nt]
            grad_obs = g[n]
            grad_iso = (ce[nt] - qn)
            tangent_iso = grad_iso @ orthogonal
            tangent_trans = tangent_iso @ Matrix(1, -r)
            grad_trans = tangent_trans @ np.linalg.inv(orthogonal)

            ax.plot(pn[0], pn[1], 'o', color='orange')
            ax.plot(qn[0], qn[1], 'o', color='red')
            ax.plot([pn[0], qn[0]], [pn[1], qn[1]], '--', color='yellow')
            l= 0.3
            ax.plot([qn[0], qn[0] + l * grad_iso[0]], [qn[1], qn[1] + l * grad_iso[1]], '-', color='red')
            ax.plot([qn[0], qn[0] + l * tangent_iso[0]], [qn[1], qn[1] + l * tangent_iso[1]], '--', color='pink')

            ax.plot([qn[0], qn[0] + l * tangent_trans[0]], [qn[1], qn[1] + l * tangent_trans[1]], '-', color='pink')

            #observed Gradient at pn
            ax.plot([pn[0], pn[0] + l*grad_obs[0]], [pn[1], pn[1] + l*grad_obs[1]], '-', color='green')

            #Transformed Constructed Gradient at pn
            ax.plot([pn[0], pn[0] + l * grad_trans[0]], [pn[1], pn[1] + l * grad_trans[1]], '-', color='pink')


        ax.plot(c[0], c[1], 'o', color='black')
        ax.plot([0], [0], '+', color='black')
        ax.set_aspect('equal')
        plt.show()
    # exit(0)


def p2c_exRS(p, c, e, r, s):
    R_1 = Matrix(1, -r)
    q = p @ R_1
    c = c @ R_1
    e = e @ R_1
    rad = radius(q, c, e)
    re = rad[:, None] * e[None, :]
    ce = c[None, :] + re
    ce2q = (ce - q) @ Matrix(s*s, r)
    return ce2q


def func_test():
    img = read_img('ecc_r_r45_Sy66.png')
    if img is None:
        exit(0)
    assert len(img.shape) == 3 and img.shape[-1] == 3
    rows, cols = img.shape[0], img.shape[1]
    cols_rows = np.array([cols, rows])
    print("Rows:{}, Cols:{}".format(rows, cols))

    # Constants
    gs = color_grads(img)
    gs_ortho = [star_grad(g) for g in gs]

    p = points([rows, cols]) / cols_rows
    origin = np.ones(2) * 0.5 # ? does this need to be trained ?
    normalized_point = p - origin[None, :]

    # Vairables
    c = np.array([0.500, 0.665] ) # origin #
    ecc = np.array([0.500, 0.665] )
    s = 0.635
    t = np.deg2rad(-45.319) #np.pi/4

    # Dependent variables
    normalized_center = c - origin

    # positional gradient are defined on each point p (in normalized domain) to have a vector G(p)
    # Reconstructed gradient
    p2c_exRS_test(normalized_point, normalized_center, ecc, t, s, g=gs[0], cols_rows=cols_rows)
    Gp = p2c_exRS(normalized_point, normalized_center, ecc, t, s)
    TxGp = Gp

    res = np.einsum('ij,ij->i', gs_ortho[0], TxGp)
    print("Residue: Sum {:.4f}, max: {:.3f}, min: {:.3f}".format(np.sum(np.abs(res)), np.max(res), np.min(res)))

    fig, ax = plt.subplots()
    draw_vector_field(gs[0], normalized_point, ax, cols_rows, color='red')
    draw_vector_field(TxGp, normalized_point, ax, cols_rows, color='blue', label="Transformed Constructed Gradient")
    plt.show()


if __name__ == '__main__':
    func_test()