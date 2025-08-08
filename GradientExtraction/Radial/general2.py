import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from hyperparam import VERBOSE


Orth = np.array([
    [0, -1],
    [1, 0]
])

I = np.array([
    [1, 0],
    [0, 1]
])

E = 1e-12 # To avoid divide by zero


def Scale(s):
    return np.array([[1,0],[0,s]])


def Rotate(theta):
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


def normalize_rad(rad):
    return (rad + np.pi) % (2 * np.pi) - np.pi


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
    gy, gx = np.gradient(color_img) #sobel_op(color_img) #
    g = np.stack([gx.flatten(), gy.flatten()], axis=1)
    mag = np.linalg.norm(g, axis=1) if normalized else np.ones(len(g))
    mag[mag < 1e-8] = 1 # convert small values to 1
    return g / mag[:, None]


def points(rows_cols):
    rows, cols = rows_cols[0], rows_cols[1]
    p = np.array(list(np.ndindex(rows, cols)))
    return np.fliplr(p)


def img_points_grad(img, crop_edges=None, normalize=True):
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
    img_g = img #gaussian_filter(img, sigma=5)

    Gs = [
        grad(img_g[:,:,i], normalized=normalize) for i in range(3)
    ]

    mask = img_g[:,:,3].flatten() if img_g.shape[2] == 4 else None
    P = points(img_g.shape[:2])
    mask3 = np.concatenate([mask, mask, mask]) if mask is not None else np.ones(3 * len(P))
    rows, cols = img_g.shape[0], img_g.shape[1]
    cols_rows = np.array([cols, rows])
    print("Image Size: Width:{}, Height:{}".format(cols_rows[0], cols_rows[1]))
    # concat color and point gradient for three channels
    Gs = np.concatenate([G for G in Gs])
    # P = np.tile(P, [3, 1])

    return img, P, Gs, mask3, cols_rows


def normalize(F):
    mag = np.linalg.norm(F, axis=1)
    mag[(mag < 1e-10)] = 1
    return F / mag[:, None]


def draw_gradient(N, G, P, ax, mask, color):
    l = 0.5
    for i, p in enumerate(P[:N]):
        if mask[i] < 1: continue
        mag = np.linalg.norm(G[i])
        if mag < 1e-10: continue
        q = p + G[i]/mag*l
        ax.plot([p[0], q[0]], [p[1], q[1]], '-', color=color)
    ax.set_aspect('equal')


def draw_Gs(cols_rows, GPs, mask):
    M = len(GPs)
    N = cols_rows[0] * cols_rows[1]
    fig, axs = plt.subplots(1, M)
    for i, GP in enumerate(GPs):
        G, P = GP
        draw_gradient(N, G, P, axs[i], mask, color='red' if i %2==0 else 'blue')
    plt.show()


def radius(PHat, fHat, eHat):
    """
        | PHat - CPHat |^2 = RPHat^2
        CPHat = fHat + eHat*RPHat
    """
    op = fHat - PHat
    a = (eHat.T @ eHat - 1)
    b = 2 * (eHat[None, :] * op).sum(axis=1)
    c = dot(op, op)
    dis = np.clip(-(4 * a * c) + (b ** 2), a_min=0, a_max=np.inf)
    return (-b - np.sqrt(dis)) / (2 * a)


class Concentric:
    def __init__(self,img):
        self.img, self.P, self.Gs, self.mask, self.cols_rows = img_points_grad(img, crop_edges=5, normalize=False)

    def fit(self, mode):
        """
        Reconstructs generalized concentric gradient parameters.
        Space J represents the observable image space. Space I  represents the iso-metric unrotated space. fHat are
        defined in I. Once a CONCENTRIC radial gradient is created in space I, we add affine transform with oHat as the origin.
        This transform the radial gradient from space I to J. We compare the gradient of the transformed gradient (which may
        not be concentric) with the observed image gradient Gs.
        :param img: Image
        :param mode: 'Fit' or 'Test'
        :return:
        """

        def fun(x):
            oHat = x[:2]
            scale = x[2]
            theta = x[3] # normalize_rad(x[3])

            # Transforms for moving points from space J (image space) to space I (isometric)
            T = Scale(np.exp(scale)) @ Rotate(theta)
            Tinv = Rotate(-theta) @ Scale(np.exp(-scale))

            Pn = self.P - oHat
            fHatn = oHat - oHat

            PHat = Pn @ Tinv
            CPHat = fHatn
            GHat1 = (PHat - CPHat) @ Orth @ T

            if mode == 'Test':
                draw_Gs(self.cols_rows, [(self.Gs, self.P), (GHat1, self.P)], self.mask)

            res = dot(self.Gs, GHat1)
            print("Residue: res:{:.2f}, oHat:{}, Scale:{:.2f}, Rotate:{:.2f}".
                  format(np.sum(np.abs(res)), np.round(oHat, 2), np.exp(scale), theta))
            return res

        def pack(o, s, r): return np.concatenate([o, [s, r]])
        if mode == 'Test':
            o = np.array([57.93, 56.52])
            scale = 1.42
            fun(x=pack(o, scale, 0))
            return self.img, None
        else:
            log_scale = 0
            theta = 0
            oHat = self.cols_rows * 0.5
            residue = least_squares(fun=fun, x0=pack(oHat, log_scale, theta), xtol=1e-10, ftol=1e-10)

            x = np.concatenate([residue.x[:2], np.zeros(2), residue.x[2:], [np.linalg.norm(self.cols_rows*0.4)]])
            return self.extract_parameters(x)

    def extract_parameters(self, x, dump=True):
        print("X: {}".format(np.round(x,3)))
        fHat = x[:2]
        scale, rotate = np.exp(x[4]), x[5]
        T = Scale(scale) @ Rotate(rotate)
        oHat = fHat
        f = (fHat - oHat) @ T + oHat

        print("--------------------------------------------------------------------------")
        print("Width Height:{}".format(self.cols_rows))
        print("fHat:{}, oHat:{}".format(fHat, oHat))
        print("Scale: {:.2f}, Rotate: {:.2f}deg".format(scale, np.rad2deg(normalize_rad(rotate))))
        print("Transform : matrix({}, {}, {}, {}, {}, {})".format(*np.round(T.flatten(), 4), 0, 0))
        print("--------------------------------------------------------------------------")

        if dump:
            plt.imshow(self.img)
            plt.plot(f[0], f[1], 'bo')
            plt.plot(oHat[0], oHat[1], 'ro')
            plt.show()
        return self.img, self.mask, self.cols_rows, fHat, np.zeros(2), oHat, 0, T, scale, rotate


class Eccentric:
    def __init__(self, image):
        self.img, self.P, self.Gs, self.mask, self.cols_rows = img_points_grad(image, crop_edges=3, normalize=False)
        self.p_scale = np.max(self.cols_rows)

    def fit(self, mode):
        """
        Reconstructs generalized radial gradient parameters in iso-metric unrotated space.
        We compare the color gradient of the created radial image with the observed image gradient Gs.
        :param mode: 'Fit' or 'Test'
        :return:
        """
        self.P = self.P / self.p_scale
        GsT = self.Gs @ Orth

        def center_Hat(PHat, fHat, eHat):
            r = radius(PHat, fHat, eHat)
            return fHat[None, :] + r[:, None] * eHat

        def fun(x):
            fHat, eHat = x[:2], x[2:4]

            PHat = self.P

            cPHat = center_Hat(PHat, fHat, eHat)
            GHat = PHat - cPHat

            GHat = np.tile(GHat, [3, 1])

            res = dot(GsT, GHat)
            print("Residue Sum:{:.2f}, fHat:{} Ecc:{}".format(np.sum(np.abs(res)), np.round(fHat, 2), np.round(eHat,2)))
            # ---------------- DEBUG ----------------
            if mode == 'Test':
                draw_Gs(self.cols_rows, [(GsT, self.P), (GHat@Orth, PHat)], self.mask)
            # ----------------------------------------
            return res

        f = np.ones(2) * 0.5 #self.cols_rows * 0.5
        e = np.zeros(2)

        def pack(o, e):
            return np.concatenate([o, e])

        if mode == 'Test':
            fun(x=pack(f, e))
            return self.img, None
        else:
            residue = least_squares(fun=fun, x0=pack(f, e), xtol=1e-10, ftol=1e-10)
            # ---------- debug -------------
            # ------------------------------
            f = residue.x[:2] * self.p_scale
            e = residue.x[2:]
            x = np.concatenate([f, e, [0, 0, 0.6]])
            return self.extract_parameters(x)

    def get_outer_radius(self, eHat, frac):
        return np.linalg.norm(self.cols_rows * frac)
        # return np.linalg.norm(eHat * cols_rows*frac)

    def fit_T(self, mode):
        """
        Reconstructs generalized radial gradient parameters.
        Space J represents the observable image space. Space I  represents the iso-metric unrotated space. fHat, eHat are
        defined in I. Once a CONCENTRIC radial gradient is created in space I, we add affine transform with oHat as the origin.
        This transform the radial gradient from space I to J. We compare the gradient of the transformed gradient (which may
        not be concentric) with the observed image gradient Gs.
        :param img: Image
        :param mode: 'Fit' or 'Test'
        :return:
        """
        self.P = self.P / self.p_scale
        ####### Fraction for outer radius #######
        frac = 0.6
        #########################################
        def fun(x):
            fHat = x[:2]
            eHat = x[2:4]
            scale, rotate = x[4], normalize_rad(x[5])
            # frac = x[6]

            T = Scale(np.exp(scale)) @ Rotate(rotate)
            Tinv = Rotate(-rotate) @ Scale(np.exp(-scale))

            outer_radius = self.get_outer_radius(eHat, frac) / self.p_scale
            oHat = fHat + outer_radius * eHat

            Pn = self.P - oHat
            fHatn = fHat - oHat

            PHat = Pn @ Tinv
            RPHat = radius(PHat, fHatn, eHat)
            CPHat = fHatn + RPHat[:, None] * eHat
            GHat1 = (PHat - CPHat) @ Orth @ T
            GHat1 = np.tile(GHat1, [3, 1])

            if mode == 'Test':
                draw_Gs(self.cols_rows, [(self.Gs, self.P), (GHat1@Orth, self.P)], self.mask)

            res = dot(self.Gs, GHat1)
            print("Residue: res:{:.2f}, fHat:{}, oHat:{}, eHat:{}, Radius:{:.4f}, Scale:{:.4f}, Rotate:{:.4f}".
                  format(np.sum(np.abs(res)), np.round(fHat, 2), np.round(oHat, 2), np.round(eHat, 2),
                         outer_radius, np.exp(scale), rotate))
            return res

        def pack(f, e, s, t, frac):
            return np.concatenate([f, e, [s, t]])

        # mode = 'Test'
        if mode == 'Test':
            fHat = np.array([15.015, 56.224])  # cols_rows * 0.5
            eHat = np.array([0.524, -0.559])  # np.zeros(2)
            s = 0.60
            t = np.deg2rad(30)
            res=fun(x = pack(fHat, eHat, s=s, t=t, frac=frac))
            print("Residue: {}".format(np.sum(np.abs(res))))
            return self.img, None
        else:
            fHat = np.ones(2) * 0.5 #self.cols_rows * 0.5
            eHat = np.zeros(2)
            s, t = 0, 0
            residue = least_squares(fun=fun, x0=pack(fHat, eHat, s=s, t=t, frac=frac), xtol=1e-5, ftol=1e-5)
            fHat = residue.x[:2] * self.p_scale
            eHat = residue.x[2:4]
            scale, rotate = residue.x[4], residue.x[5]
            if len(residue.x) == 7:
                frac = residue.x[6]
            x = np.concatenate([fHat, eHat, [scale, rotate, frac]])
            return self.extract_parameters(x=x)
            # return self.extract_parameters(x=None)

    def extract_parameters(self, x, dump=True):
        # x = np.array([3.93816e+02, -7.96690e+01, -5.31000e-01,  5.75000e-01, -3.03000e-01, 2.67558e+02,  6.00000e-01])
        # x = np.array([1.07593e+02,  3.21101e+02, -2.83000e-01, -9.82000e-01,  5.89000e-01, 3.08421e+02, 6.00000e-01])
        print("X:", np.round(x,3))
        fHat, eHat = x[:2], x[2:4]
        scale, rotate = np.exp(x[4]), x[5]
        Ro = self.get_outer_radius(eHat, x[6])
        T = Scale(scale) @ Rotate(rotate)
        oHat = fHat + eHat * Ro
        f = (fHat - oHat) @ T + oHat

        print("--------------------------------------------------------------------------")
        print("Width Height:{}, Outer radius:{}, fraction:{}".format(self.cols_rows, Ro, x[6]))
        print("fHat:{}, oHat:{}".format(fHat, oHat))
        print("Radius:{:.2f}, Scale: {:.2f}, Rotate: {:.2f}deg".format(Ro, scale, np.rad2deg(normalize_rad(rotate))))
        print("Transform : matrix({}, {}, {}, {}, {}, {})".format(*np.round(T.flatten(), 4), 0, 0))
        print("--------------------------------------------------------------------------")

        if dump:
            plt.imshow(self.img)
            plt.plot(f[0], f[1], 'bo')
            plt.plot(oHat[0], oHat[1], 'ro')
            plt.title("focus and center")
            # set axis off
            plt.gca().axis('off')
            plt.show()
            # plt.savefig("f_c.png", dpi=300)
            plt.close()
        return self.img, self.mask, self.cols_rows, fHat, eHat, oHat, Ro, T, scale, rotate


class Alternate:
    def get_outer_radius_(x6):
        return abs(x6)
        # return np.exp(x6)
        # return (frac)**2

    def fit_ecc_T_(self, img, mode):
        """
        Reconstructs generalized radial gradient parameters.
        Space J represents the observable image space. Space I  represents the iso-metric unrotated space. fHat, eHat are
        defined in I. Once a CONCENTRIC radial gradient is created in space I, we add affine transform with oHat as the origin.
        This transform the radial gradient from space I to J. We compare the gradient of the transformed gradient (which may
        not be concentric) with the observed image gradient Gobs.
        :param img: Image
        :param mode: 'Fit' or 'Test'
        :return:
        """
        img, P, Gobs, mask, cols_rows = img_points_grad(img=img, crop_edges=5, normalize=False)

        def radius(PHat, fHat, eHat):
            op = fHat - PHat
            a = (eHat.T @ eHat - 1)
            b = 2 * (eHat[None, :] * op).sum(axis=1)
            c = dot(op, op)
            dis = np.clip(-(4 * a * c) + (b ** 2), a_min=0, a_max=np.inf)
            return (-b - np.sqrt(dis)) / (2 * a + E)

        def fun(x):
            fHat = x[:2]
            oHat = x[2:4]
            scale, rotate = x[4], normalize_rad(x[5])
            outer_radius = Alternate.get_outer_radius_(x[6])

            eHat = (-fHat + oHat) / outer_radius
            # oHat = fHat + outer_radius * eHat

            T = Scale(np.exp(scale)) @ Rotate(rotate)
            Tinv = Rotate(-rotate) @ Scale(np.exp(-scale))
            # T = I
            # Tinv = I
            Pn = P - oHat
            fHatn = fHat - oHat

            PHat = Pn @ Tinv
            RPHat = radius(PHat, fHatn, eHat)
            CPHat = fHatn + RPHat[:, None] * eHat
            GHat1 = (PHat - CPHat) @ Orth @ T

            if mode == 'Test':
                draw_Gs(cols_rows, [(Gobs, P), (GHat1@Orth, P)], mask)

            res = dot(Gobs, GHat1)
            print("Residue: res:{:.2f}, fHat:{}, oHat:{}, Radius:{:.2f}, Scale:{:.2f}, Rotate:{:.2f}".
                  format(np.sum(np.abs(res)), np.round(fHat, 2), np.round(oHat, 2), outer_radius, np.exp(scale), rotate))
            return res

        def pack(f, o, s, t, r):
            return np.concatenate([f, o, [s, t, r]])

        mode = 'Test'
        if mode == 'Test':
            oHat = np.array([11.99, 35.57]) #cols_rows*0.5
            fHat = np.array([10.59, 35.57]) #np.array([11, cols_rows[1]*0.5])
            s, t = np.log(.7), 0
            r = 35 #cols_rows[1]*0.5  # np.log(np.linalg.norm(cols_rows)*0.5)
            res=fun(x = pack(fHat, oHat, s=s, t=t, r=r))
            Alternate.extract_parameters_(img, cols_rows, [])
            print("Residue: {}".format(np.sum(np.abs(res))))
            return img, None
        else:
            oHat = cols_rows * 0.5
            fHat = np.array([11, cols_rows[1] * 0.5])
            s, t = np.log(.7), 0
            r = cols_rows[1] * 0.5  # np.log(np.linalg.norm(cols_rows)*0.5)
            res = fun(x=pack(fHat, oHat, s=s, t=t, r=r))
            print("***** Residue: {}".format(np.sum(np.abs(res))))
            print("**********************************************")
            residue = least_squares(fun=fun, x0=pack(fHat, oHat, s=s, t=t, r=r), xtol=1e-5, ftol=1e-5)
            return Alternate.extract_parameters_(img, cols_rows, residue.x)

    def extract_parameters_(img, cols_rows, x, dump=True):
        fHat, oHat = x[:2], x[2:4]
        scale, rotate = np.exp(x[4]), x[5]
        frac = x[6] if len(x) ==7 else 1
        T = Scale(scale) @ Rotate(rotate)
        Ro = Alternate.get_outer_radius_(frac)
        f = (fHat - oHat) @ T + oHat
        print("--------------------------------------------------------------------------")
        print("Width Height:{} Outer radius frac: {:.4f}".format(cols_rows, frac))
        print("Focal:{}, Center:{}".format(fHat, oHat))
        print("Radius:{:.2f}, Scale: {:.2f}, Rotate: {:.2f}deg".format(Ro, scale, np.rad2deg(normalize_rad(rotate))))
        print("Transform : matrix({}, {}, {}, {}, {}, {})".format(*np.round(T.flatten(), 4), 0, 0))
        print("--------------------------------------------------------------------------")

        if dump:
            plt.imshow(img)
            plt.plot(f[0], f[1], 'bo')
            plt.plot(oHat[0], oHat[1], 'ro')
            plt.show()

            from Draw.svg import create_dummy_color_radial_grad
            create_dummy_color_radial_grad(cols_rows=cols_rows, origin=oHat, focal=fHat, outer_radius=Ro, T=T)
        return img, fHat, oHat, oHat, Ro, T


class StopExtractor:
    # ---------------- Stop Extraction --------------------
    def __init__(self, img, mask, fHat, eHat, scale, rotate, Ro):
        self.img = img
        self.mask = mask
        self.fHat = fHat
        self.eHat = eHat
        self.scale = scale
        self.rotate = rotate
        self.Ro = Ro
        self.T = Scale(self.scale) @ Rotate(self.rotate)
        self.Tinv = Rotate(-self.rotate) @ Scale(1 / self.scale)
        self.oHat = self.fHat + self.eHat * self.Ro

    def index(self, point):
        point = np.round(point)
        return int(point[1]), int(point[0])

    def weighted_circular_avg(self, p, N=50, ax=None):
        def weights(n, frac=None):
            m = n if frac is None else int(np.ceil(n*frac))
            assert m <= n
            if m < 1 : return 0
            a = np.arange(m, 0, -1)**2
            if n > m: a = np.concatenate([np.zeros(n-m), a])
            return a / np.sum(a)

        def in_bounds(ij):
            i,j = ij
            return 0 <= i < self.img.shape[0] and 0 <= j < self.img.shape[1]

        def avg_color(ts):
            half_circ = cp + rp * np.stack([np.cos(ts), np.sin(ts)]).T
            half_circ_T = (half_circ - self.oHat) @ self.T + self.oHat

            if ax is not None:
                ax.plot(half_circ_T[:, 0], half_circ_T[:, 1], '--', color='aqua', alpha=0.5)

            colors = []
            for p in half_circ_T:
                pij = self.index(p)
                if not in_bounds(pij) or self.mask[pij] < 1: continue
                colors.append(self.img[pij])
            colors = np.array(colors)
            if len(colors) == 0: return  None
            w = weights(len(colors), 0.7)
            return np.sum(w[:, None] * colors, axis=0)

        pHat = (p - self.oHat) @ self.Tinv + self.oHat
        rp = radius(pHat[None, :], self.fHat, self.eHat)
        cp = self.fHat + rp * self.eHat

        counter_clock_weighted_avg = avg_color(ts = np.arange(0, np.pi+np.pi / N, np.pi / N))
        clock_weighted_avg = avg_color(ts = np.arange(0, -np.pi- np.pi / N, - np.pi / N))
        if counter_clock_weighted_avg is None:
            return clock_weighted_avg
        elif clock_weighted_avg is None:
            return counter_clock_weighted_avg
        else:
            return (counter_clock_weighted_avg + clock_weighted_avg) * 0.5

    def get_color_profile(self, start, direction, max_length):
        profile = []
        coords = []
        ax = None
        if VERBOSE:
            fig, ax = plt.subplots(tight_layout=True)
            ax.set_aspect('equal')
            ax.imshow(self.img)

        for t in range(max_length):
            p = start + t * direction
            axis = ax if VERBOSE and t % 25 == 0 else None
            avg_color_p = self.weighted_circular_avg(p, ax=axis)
            if avg_color_p is not None:
                profile.append(avg_color_p)
                coords.append(self.index(p))
        if VERBOSE:
            ax.axis('off')
            plt.title("Colour Contour")
            plt.show()
            # print("Color Profile Length:{}".format(len(profile)), ", Saving at: color_contour.png")
            # plt.savefig("color_contour.png", dpi=300)
            plt.close()
        return np.array(profile), np.array(coords)

    def stop_extractor(self, start, direction):
        from misc import laplacian1d
        from StopExtraction import get_peaks, cluster, co_linearity
        max_length = int(np.round(np.linalg.norm(self.img.shape[:2]))*1.5)
        profile, coords = self.get_color_profile(start, direction, max_length)

        bumps = [np.abs(laplacian1d(profile[:, i])) for i in range(3)]
        peaks = [get_peaks(bumps[i]) for i in range(3)]

        if VERBOSE:
            fig, axs = plt.subplots(3, 3)
            x = np.arange(0, len(profile))
            for i in range(3):
                axs[0, i].plot(x, profile[:,i])
                axs[1, i].plot(x, bumps[i])
                peak = np.array(peaks[i]).astype(int)
                f_peak = cluster(bumps[i], peak)
                # print("{}\n\t|_{}".format(peak, f_peak))
                axs[1, i].plot(peak, bumps[i][peak], 'o', color='red')
                axs[1, i].plot(f_peak, bumps[i][f_peak], 'o', color='blue')

        peaks = np.sort(list(set(cluster(bumps[0], peaks[0]) + cluster(bumps[1], peaks[1]) + cluster(bumps[2], peaks[2]))))
        if VERBOSE: print("Peaks ", peaks)
        threshold = 0.005 * np.max(bumps)
        peaks = co_linearity(profile, peaks, thresold=threshold)
        if VERBOSE: print("Threshold:{}, Peaks:{} ".format(threshold, peaks))

        if VERBOSE:
            x = np.arange(0, len(profile))
            for i in range(3):
                axs[2, i].plot(x, bumps[i])
                axs[2, i].plot(peaks, bumps[i][peaks], 'o', color='red')
            fig.suptitle("Threshold:{}, Peaks:{} ".format(threshold, peaks))
            plt.show()

        return profile[peaks], coords[peaks]

    def build_svg(self):
        start = (self.fHat - self.oHat) @ self.T + self.oHat
        direction = (self.oHat - start)
        direction_norm = np.linalg.norm(direction)
        if direction_norm < E:
            direction = np.array([1,0])
        else:
            direction = direction / direction_norm

        profile, coord_indicies = self.stop_extractor(start, direction)

        p = np.flip(coord_indicies[-1])
        pHat = (p - self.oHat)  @ self.Tinv + self.oHat
        outer_radius = radius(pHat[None, :], self.fHat, self.eHat).squeeze()
        outer_center = self.fHat + self.eHat*outer_radius

        diff = np.linalg.norm(coord_indicies[1:] - coord_indicies[:-1], axis=1)
        diff = np.insert(diff, 0, 0)
        assert len(diff) == len(profile)
        length = np.cumsum(diff)
        length = length / length[-1]

        if VERBOSE:
            plt.imshow(self.img)
            plt.plot(self.oHat[0], self.oHat[1], "+")
            plt.plot(start[0], start[1], "*")
            plt.plot([start[0], start[0] + 100*direction[0]], [start[1], start[1] + 100*direction[1]], '--')
            plt.plot(coord_indicies[:,1], coord_indicies[:,0], '--')
            plt.plot(p[0], p[1], 'go')
            plt.plot(outer_center[0], outer_center[1], 'g*')
            plt.plot(coord_indicies[:,1], coord_indicies[:,0], 'go--')
            plt.title("Outer radius: {:.2f}".format(outer_radius))
            plt.show()

        from Draw.svg import create_radial_grad_ecc_T
        cols_rows = np.flip(np.array(self.img.shape[:2]))
        create_radial_grad_ecc_T(cols_rows=cols_rows, oHat=self.oHat, fHat=self.fHat, outer_center=outer_center, outer_radius=outer_radius, T=self.T, stops=(profile, length))


def reconstruct_and_save_svg(img, mode):
    if mode == 'Test':
        Eccentric(img).fit_T(mode)
    else:
        img, mask, cols_rows, fHat, eHat, oHat, Ro, T, scale, rotate = Eccentric(img).fit_T(mode)
        if np.linalg.norm(eHat) > 1:
            img, mask, cols_rows, fHat, eHat, oHat, Ro, T, scale, rotate = Eccentric(img).fit(mode)

        # from Draw.svg import create_dummy_color_radial_grad
        # create_dummy_color_radial_grad(cols_rows=cols_rows, origin=oHat, focal=fHat, outer_radius=Ro, T=T)
        # print("Mask shape:", mask.shape)
        mask = mask.reshape(img.shape[0], img.shape[1], 3)[:, :, 0]
        StopExtractor(img=img, mask=mask, fHat=fHat, eHat=eHat, Ro=Ro, scale=scale, rotate=rotate).build_svg()


def main(file):
    from helper import read_img
    img = read_img('General', file + '.png', crop_edges=False)
    reconstruct_and_save_svg(img, mode='Fit')


if __name__ == '__main__':
    # main('test_img')
    # main('Ankit/Asset 2')
    # main("Con200")
    img_path = "/Users/shaliniveesamsetty/Desktop/rad2.png"
    from PIL import Image
    img = np.array(Image.open(img_path)) / 255.
    print(img.shape, "Range:", np.min(img), np.max(img))
    reconstruct_and_save_svg(img, mode='Fit')

