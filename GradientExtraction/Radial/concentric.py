import numpy as np
from ellipse3 import star_grad, points, plt
from hyperparam import VERBOSE
from scipy.optimize import least_squares
from tools import timer
import StopExtraction as stops_extractor
from Draw.draw import draw_vector_field
from Radial.helper import color_grads, read_img


@timer
def least_square_solve(gs, P, cols_rows, initial_c=None, nomralize = False):
    gs_ortho = np.concatenate([star_grad(g) for g in gs])
    x0 = cols_rows * 0.5 if initial_c is None else initial_c

    def optimize(x):
        constructed_grad = np.tile(P - x[None, :], [len(gs), 1])
        if nomralize:
            mag = np.linalg.norm(constructed_grad, axis=1)
            mag[mag< 1e-5] = 1
            constructed_grad = np.einsum('i, ij->ij' ,(1/mag), constructed_grad)
        return np.einsum('ij,ij->i', gs_ortho, constructed_grad)

    residue = least_squares(optimize, x0=x0, xtol=1e-10, ftol=1e-10)
    res = optimize(residue.x)
    print("\nResidue: Sum {:.4f}, max: {:.3f}, min: {:.3f}".format(np.sum(np.abs(res)), np.max(res), np.min(res)))
    return residue.x


def linear_solve(gs, P, cols_rows):
    P = np.tile(P, [len(gs), 1])
    G = np.concatenate([g for g in gs])
    Gy2, Gx2, GxGy = G[:,1]**2, G[:,0]**2, G[:,0]*G[:,1]
    A, B = - np.sum(Gy2), np.sum(GxGy)
    C, D = B, -np.sum(Gx2)
    E, F = np.sum(GxGy*P[:,1]) - np.sum(Gy2*P[:,0]), np.sum(GxGy*P[:,0]) - np.sum(Gx2*P[:,0])
    M = np.array([
        [A, B],
        [C, D]
    ])
    assert np.linalg.det(M) != 0
    Minv = np.linalg.inv(M)
    x = np.array([E, F]) @ Minv
    return x


def reconstruction(img, REGRESSION_MODE, normalize=False):
    # Constants
    gs = color_grads(img)
    rows, cols = img.shape[0], img.shape[1]
    cols_rows = np.array([cols, rows])
    P = points([rows, cols])

    if VERBOSE and not REGRESSION_MODE:
        print("Rows:{}, Cols:{}".format(rows, cols))
        fig, ax = plt.subplots(1,3, figsize=(15, 10))
        colors = ['red', 'green', 'blue']
        for i in range(3):
            draw_vector_field(gs[i], P, ax[i], cols_rows, color=colors[i], label=colors[i])
            ax[i].imshow(img)
        plt.show()
    gs = [gs[1]]
    # 1. Closed form linear solve
    # return linear_solve(gs, P, cols_rows)

    # 2. Least square optimization
    lse_center= least_square_solve(gs, P, cols_rows, cols_rows*0.5, normalize)
    # if VERBOSE and not REGRESSION_MODE:
    #     fig, axs = plt.subplots()
    #     P2C = lse_center[None, :] - P
    #     G0 = gs[0]
    #     draw_vector_field(G0, P, axs, cols_rows, color='red')
    #     draw_vector_field(P2C, P, axs, cols_rows, color='blue')
    #     plt.show()
    return lse_center


def test():
    file_center_times_residue_normalize =[
        ['Con2Stops', [70.750, 72.939], 0.0221, 292.9233, False],
        ['Con2Stops', [70.744, 72.935], 0.1120, 5.0708, True],
        ['Con3Stops', [191.581, 149.775], 0.0234, 1208.4851, False],
        ['con_t', [7.336, 6.152], 0.0023, 1.5664, False],
        ['Con_simple2', [124.000, 124.000], 0.0134, 3531.9288, False],
        ['Con_Real1', [121.729, 101.409], 0.3731, 104.2538, True],
        ['Con_Real2', [243.992, 660.746], 3.3578, 188.2951, True],
        ['Con_Real3', [249.548, 531.213], 1.8154, 194.7187, True],
        ['Con_Real4', [201.989, 689.292], 7.4840, 183.5095, True],
        ['Con_Center_Outside', [122.914, -16.693], 0.2047, 7.9775, True],
        ['Con_Center_Outside', [122.920, -15.945], 0.0222, 734.2382, False]
    ]

    for name, c, t, res, n in file_center_times_residue_normalize:
        img = read_img('Concentric', name + ".png")
        center = reconstruction(img, REGRESSION_MODE=True, normalize=n)
        print("File:{}, Normalized: {}\n"
              "------------------------------------------------"
              "\n\tConstructed Center : [{:.3f}, {:.3f}],"
              "\n\tRecorded Center    : [{:.3f}, {:.3f}],"
              "\n\tRecorded time      : {}s"
              "\n\tRecorded Residue   : {}\n"
              "------------------------------------------------".
              format(name, 'Y' if n else 'N', center[0], center[1], c[0], c[1], t, res))
        assert np.linalg.norm(center - c) <= 1


def to_svg(img, center, stop_colors, stop_coords, name):
    from Draw.svg import create_concentric_radial
    from StopExtraction import stop_percentage, index
    center_coord = np.array(index(center))
    radius = np.linalg.norm(center_coord-stop_coords[-1])
    create_concentric_radial(img.shape, center, radius, stop_percentage(center_coord, stop_coords), stop_colors, name)


if __name__ == '__main__':
    name = 'Con_Real2'
    img = read_img('Concentric', name + '.png', crop_edges=True)
    print("Image size:", img.shape)
    center = reconstruction(img, REGRESSION_MODE=False, normalize=True)
    if VERBOSE:
        print("-------------------------------------"
              "\n\tCenter \t\t :[{:.3f}, {:.3f}]\n"
              "-------------------------------------".
              format(center[0], center[1]))
        fig, axs = plt.subplots(figsize=(15, 10))
        axs.imshow(img)
        axs.plot(center[0], center[1], '*')
        plt.show()
    colors, stop_coords = stops_extractor.concentric(img, center, mask=None)
    to_svg(img, center,  colors, stop_coords, name)
    if VERBOSE:
        print("-------------------------------------"
              "\n\tCenter \t\t :[{:.3f}, {:.3f}]\n"
              "-------------------------------------".
              format(center[0], center[1]))
        fig, axs = plt.subplots(figsize=(15, 10))
        axs.imshow(img)
        axs.plot(stop_coords[:,1], stop_coords[:,0], 'o--')
        axs.plot(center[0], center[1], '*')
        plt.show()