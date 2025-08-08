import numpy as np
from scipy.optimize import least_squares
from tools import timer
from hyperparam import VERBOSE
from Radial.helper import color_grads, read_img, radius
from ellipse3 import star_grad, points, plt
import StopExtraction as stops_extractor


def center(p, o, e):
    r = radius(p, o, e)
    return o[None, :] + r[:, None]*e[None, :]


@timer
def least_square_solve(gs, P, cols_rows, initial_c=None, nomralize = False):
    gs_ortho = np.concatenate([star_grad(g) for g in gs])
    c0 = cols_rows * 0.5 if initial_c is None else initial_c
    ecc = np.zeros(2)

    def pack(c, e):
        return np.concatenate([c, e])

    def unpack(x):
        return x[:2], x[2:4]

    def optimize(x):
        o, e = unpack(x)
        constructed_grad = np.tile(P - center(P, o, e), [3, 1])
        if nomralize:
            mag = np.linalg.norm(constructed_grad, axis=1)
            mag[mag< 1e-5] = 1
            constructed_grad = np.einsum('i, ij->ij' ,(1/mag), constructed_grad)
        return np.einsum('ij,ij->i', gs_ortho, constructed_grad)

    residue = least_squares(optimize, x0=pack(c0, ecc), xtol=1e-10, ftol=1e-10)
    res = optimize(residue.x)
    print("\nResidue: Sum {:.4f}, max: {:.3f}, min: {:.3f}".format(np.sum(np.abs(res)), np.max(res), np.min(res)))
    return unpack(residue.x)


def reconstruction(img, REGRESSION_MODE, normalize=False):
    # Constants
    gs = color_grads(img)
    rows, cols = img.shape[0], img.shape[1]
    cols_rows = np.array([cols, rows])
    P = points([rows, cols])

    # if VERBOSE and not REGRESSION_MODE:
    #     print("Rows:{}, Cols:{}".format(rows, cols))
    #     fig, ax = plt.subplots(1,3)
    #     colors = ['red', 'green', 'blue']
    #     for i in range(3):
    #         draw_vector_field(gs[i], P, ax[i], cols_rows, color=colors[i], label=colors[i])
    #         ax[i].imshow(img)
    #     plt.show()

    lse_center, lse_ecc = least_square_solve(gs, P, cols_rows, None, normalize)
    # if VERBOSE and not REGRESSION_MODE:
    #     fig, axs = plt.subplots()
    #     P2C = lse_center[None, :] - P
    #     G0 = gs[0]
    #     draw_vector_field(G0, P, axs, cols_rows, color='red')
    #     draw_vector_field(P2C, P, axs, cols_rows, color='blue')
    #     plt.show()
    return lse_center, lse_ecc


def to_svg(img, center, ecc, stop_colors, stop_coords, name):
    from Draw.svg import create_eccentric_radial
    from StopExtraction import stop_percentage, index
    center_coord = np.array(index(center))
    p = np.array([stop_coords[-1, 1], stop_coords[-1, 0]])
    r = radius(p.reshape(1,-1), center, ecc).squeeze()
    c_p = center + ecc * r
    print("Center :{}\nRadius :{}\n"
          "Eccentricity:{} \nStop percentage :{}".format(center, r, ecc,
                                                         stop_percentage(center_coord, stop_coords)))
    # if VERBOSE:
    #     N=100
    #     l = np.arange(0, 2*np.pi, 2*np.pi/N)
    #     fig, ax = plt.subplots()
    #     ax.imshow(img)
    #     # r = np.linalg.norm(c-p)
    #     ax.plot(c_p[0], c_p[1], 'o')
    #     ax.plot(c_p[0] + r*np.cos(l), c_p[1] + r*np.sin(l), '--', color='yellow')
    #     plt.show()
    create_eccentric_radial(img.shape, center, c_p,
                            radius=r,
                            stop_percentage=stop_percentage(center_coord, stop_coords),
                            stop_colors=stop_colors, name=name)


if __name__ == '__main__':
    name= 'Ecc_d'
    img = read_img('Eccentric', name+".png", crop_edges=True)
    center, ecc = reconstruction(img, REGRESSION_MODE=False, normalize=False)
    colors, stop_coords = stops_extractor.eccentric(img, center, ecc, mask=None)
    to_svg(img, center, ecc, colors, stop_coords, name=name)
    if VERBOSE:
        print("-------------------------------------"
              "\n\tCenter \t\t :[{:.3f}, {:.3f}]\n"
              "-------------------------------------".
              format(center[0], center[1]))
        fig, axs = plt.subplots()
        axs.imshow(img)
        axs.plot(stop_coords[:,1], stop_coords[:,0], 'o--')
        axs.plot(center[0], center[1], '*')
        plt.show()