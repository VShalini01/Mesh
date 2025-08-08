import torch as th
import torch.nn.functional as F
import numpy as np
from torch import nan_to_num
import scipy.fftpack
import imutils
import statistics
from GradientExtraction.Radial.hyperparam import ERROR, VERBOSE
import matplotlib.pyplot as plt


sobel_x = th.Tensor([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
sobel_y = th.Tensor([
    [ 1,  2,  1],
    [ 0,  0,  0],
    [-1, -2, -1]
])
gaussian = th.Tensor([
    [1,  4,  7,  4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1,  4,  7,  4, 1]
]) / 273
laplacian_filter = th.Tensor([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])
log_filter = th.Tensor([
    [0,  1,  1,  2,  2,  2,  1,  1,  0],
    [1,  2,  4,  5,  5,  5,  4,  2,  1],
    [1,  4,  5,  3,  0,  3,  5,  4,  1],
    [2,  5,  3,-12,-24,-12,  3,  5,  1],
    [2,  5,  0,-24,-40,-24,  0,  5,  2],
    [2,  5,  3,-12,-24,-12,  3,  5,  2],
    [1,  4,  5,  3,  0,  3,  5,  4,  1],
    [1,  2,  4,  5,  5,  5,  4,  2,  1],
    [0,  1,  1,  2,  2,  2,  1,  1,  0]
])


def convolute2D(m, f):
    assert len(m.shape) == 2 and len(f.shape) == 2 and f.shape[0] == f.shape[1] and f.shape[0] % 2 != 0
    m1 = m.reshape(1, 1, m.shape[0], m.shape[1])
    c1 = f.reshape(1, 1, f.shape[0], f.shape[1])
    padding = int(f.shape[0]/2)
    out = F.conv2d(m1, c1, stride=1, padding=padding)
    return out[0,0]


def convolute1D(m, f):
    assert len(m.shape) == 1 and len(f.shape) == 1
    m1 = m.reshape(1, 1, m.shape[0])
    c1 = f.reshape(1, 1, f.shape[0])
    padding = int(f.shape[0] / 2)
    out = F.conv1d(m1, c1, stride=1, padding=padding)
    return out[0,0]


def grad(img, alpha_mask, normalize=True):
    assert len(img.shape) == 2
    Gy, Gx = np.gradient(tensor2numpy(img)) #convolute2D(img, sobel_x), convolute2D(img, sobel_y)
    Gx, Gy = th.Tensor(Gx), -1*th.Tensor(Gy)
    mag = th.sqrt(Gx**2 + Gy**2)
    if alpha_mask is not None: mag = mag * alpha_mask
    mask = (mag >= 1e-5).type(th.int)
    if normalize:
        Gx, Gy = mask*nan_to_num(Gx/mag, nan=0, posinf=0, neginf=0), mask*nan_to_num(Gy/mag, nan=0, posinf=0, neginf=0)
    else:
        Gx, Gy = mask*nan_to_num(Gx, nan=0, posinf=0, neginf=0), mask*nan_to_num(Gy, nan=0, posinf=0, neginf=0)
    return Gx, Gy


def gaussian_smoothen(img):
    assert len(img.shape)==2
    return convolute2D(img, gaussian)


def laplacian(img):
    assert len(img.shape) == 2
    return convolute2D(img, laplacian_filter)


def log(img):
    assert len(img.shape) == 2
    return convolute2D(img, log_filter)


def tensor2numpy(tensor):
    if th.is_tensor(tensor):
        return tensor.cpu().numpy() if th.cuda.is_available() else tensor.numpy()
    else:
        return tensor


def low_pass_filter(y):
    assert len(y.shape) == 1
    y = tensor2numpy(y)
    x = np.arange(0, y.shape[0], 1)
    w = scipy.fftpack.rfft(y)
    # f = scipy.fftpack.rfftfreq(len(y), x[1] - x[0])
    spectrum = w ** 2

    cutoff_idx = spectrum < (spectrum.max() / 20)
    w2 = w.copy()
    w2[cutoff_idx] = 0

    return th.Tensor(scipy.fftpack.irfft(w2))


def double_diff(s, support=3):
    assert support > 1
    left = th.ones(support-1)#th.arange(0, support, 1)
    right = th.ones(support-1)#th.arange(support-1, -1, -1)
    filter = th.cat((-left, th.Tensor([2*len(left)]), -right))
    return convolute1D(s, filter)


def gaussain_smoothen1d(series):
    to_revert = False
    if not th.is_tensor(series):
        series = th.Tensor(series)
        to_revert = True
    filter = th.Tensor([0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006])
    p_series = th.cat([series[0]*th.ones(3), series, series[-1]*th.ones(3)])
    con = convolute1D(p_series, filter)
    return tensor2numpy(con[3:-3]) if to_revert else con[3:-3]


def avg_smoothen1d(series, size=13):
    to_revert = False
    if not th.is_tensor(series):
        series = th.Tensor(series)
        to_revert = True

    filter = th.ones(2*size+1)/(2*size+1)
    p_series = th.cat([series[0]*th.ones(size), series, series[-1]*th.ones(size)])
    con = convolute1D(p_series, filter)
    return tensor2numpy(con[size:-size]) if to_revert else con[size:-size]


def laplacian1d(series, size=21):
    to_revert = False
    if not th.is_tensor(series):
        series = th.Tensor(series)
        to_revert = True

    left = th.ones(size)
    right = th.ones(size)
    filter = th.cat((-left, th.Tensor([2 * size]), -right))
    p_series = th.cat([series[0] * th.ones(size), series, series[-1] * th.ones(size)])
    con = convolute1D(p_series, filter)
    return tensor2numpy(con[size:-size]) if to_revert else con[size:-size]


def normalize_angle(rad):
    deg = np.rad2deg(rad)
    time = int(deg/360)
    return np.deg2rad(deg - time*360)


def tensor_img(img):
    assert len(img.shape) == 3 and img.shape[2] == 3
    img = img / 255
    img = th.stack([th.Tensor(img[:, :, 0]), th.Tensor(img[:, :, 1]), th.Tensor(img[:, :, 2])])
    assert img.shape[0] == 3
    return img


def smoothen(img):
    assert len(img.shape) == 3 and img.shape[0] == 3
    return th.stack([gaussian_smoothen(img[i]) for i in range(img.shape[0])])


def sanitize(image_tensor):
    img = np.zeros(shape=(image_tensor.shape[1], image_tensor.shape[2], 3))
    if image_tensor.shape[0] == 3:
        for i in range(3):
            img[:, :, i] = tensor2numpy(image_tensor[i])
    else:
        img = tensor2numpy(image_tensor)
    return img


def rotate(image_tensor, radian):
    r = int(np.sqrt(image_tensor.shape[1]**2 + image_tensor.shape[2]**2)/2)
    w, h = r - int(image_tensor.shape[1]*0.5), r - int(image_tensor.shape[2]*0.5)
    img = np.pad(sanitize(image_tensor), [(w, w),(h, h), (0, 0)], mode='constant', constant_values=0)
    img = imutils.rotate(img, np.rad2deg(radian))
    return tensor_img(img)


def rotate_with_mask(image_tensor, mask, radian):
    """ Return numpy image of shape HxWx4"""
    r = int(np.sqrt(image_tensor.shape[1] ** 2 + image_tensor.shape[2] ** 2) / 2)
    w, h = r - int(image_tensor.shape[1] * 0.5), r - int(image_tensor.shape[2] * 0.5)
    r_img = np.zeros((image_tensor.shape[1], image_tensor.shape[2], 4))
    r_img[:,:,0] = tensor2numpy(image_tensor[0])
    r_img[:,:,1] = tensor2numpy(image_tensor[1])
    r_img[:,:,2] = tensor2numpy(image_tensor[2])
    r_img[:,:,3] = tensor2numpy(mask)
    pad = [(w, w), (h, h), (0, 0)]
    r_img = np.pad(r_img, pad, mode='constant', constant_values=0)
    r_img = imutils.rotate(r_img, np.rad2deg(radian))
    return r_img, pad


def rotate_without_pad(image, radian):
    """ Return numpy image of shape HxWx4"""
    r_img = imutils.rotate(image, np.rad2deg(radian))
    return r_img


def principal_axis(points):
    c_poly = points - np.mean(points, axis=0)
    u, s, vt = np.linalg.svd(c_poly / np.sqrt(len(points)), full_matrices=False)
    return vt, s


def rolling_window_cluster(candidates, window_size):
    winners = []
    window = []
    for i in range(len(candidates)):
        if len(window) == 0:
            window.append(candidates[i])
        elif len(window) > 0 and window[-1] + window_size > candidates[i]:
            window.append(candidates[i])
        else:
            winners.append(int(statistics.median(window)))
            window = [candidates[i]]

    if len(window) > 0:
        winners.append(int(statistics.median(window)))
    return winners


def exterior_angle(p0, p1, p2):
    """ returns the exterior angle in degrees defined byt the lines po--p1--p2"""
    p0, p1, p2 = np.array(p0), np.array(p1), np.array(p2)
    p10 = p1 - p0
    p21 = p2 - p1
    l1, l2 = np.linalg.norm(p10), np.linalg.norm(p21)
    if l1*l2 == 0:
        return 0
    else:
        d = np.clip(p10@p21/(l1*l2), -1, 1)
        return np.rad2deg(np.arccos(d))


def rotate_point(point, radian, origin):
    R = np.array([
        [np.cos(radian), -np.sin(radian)],
        [np.sin(radian),  np.cos(radian)]
    ])
    return origin + R @ (point - origin)


def extend2boundary(shape, point, theta):
    H, W = shape[0], shape[1]
    x, y = point[0], point[1]

    right, left = None, None

    return left, right


def g_smoothen(series, filter_length = 10):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(series, filter_length)


def get_boundary(rows, cols, point, dx, dy):
    """Returns start and end of the axis line"""
    px, py = int(point[0]), int(point[1])
    if dx == 0 and dy == 0:
        return None

    theta = np.arctan2(dy, dx)
    corner = np.arctan2(rows - point[1], cols - point[0])

    if -np.sin(corner) < np.sin(theta) < np.sin(corner):
        q0y = point[1] + dy*(-point[0]/dx)
        q1y = point[1] + dy*(cols - point[0])/dx
        start, end = np.array([0, q0y]), np.array([cols, q1y])
    else:
        q0x = point[0] + dx*(-point[1])/dy
        q1x = point[0] + dx*(rows - point[1])/dy
        start, end = np.array([q0x, 0]), np.array([q1x, rows])

    return np.array([start, end])


def distance(p, line):
    s, e = line
    assert np.linalg.norm(e-s) > 0
    if np.linalg.norm(p-s) == 0:
        return 0

    c = (p-s)@(e-s) / (np.linalg.norm(p-s)*np.linalg.norm(e-s))
    theta = np.arccos(np.clip(c, -1, 1))
    return np.sin(theta) * np.linalg.norm(p-s)


def closest_point(p, line):
    s, e = line
    assert np.linalg.norm(e - s) > 0
    c1 = (p-s)@(e-s) / np.linalg.norm(e-s)
    c2 = (p-e)@(s-e) / np.linalg.norm(e-s)
    return c2 * s / (c1+c2) + c1 * e / (c1+c2)


def filter_color_stops(stop_indices, colors, co_linear_threshold=1e-2):
    """Filter out color/stops if the consecutive colors are co-linear"""
    filtered_stops = [stop_indices[0]]
    filtered_colors = [colors[0]]
    for i in range(1, len(colors) - 1):
        if np.linalg.norm(filtered_colors[-1] - colors[i + 1]) <= 1e-10: continue
        co_linearity = (np.linalg.norm(colors[i] - filtered_colors[-1]) + np.linalg.norm(
            colors[i] - colors[i + 1])) / np.linalg.norm(filtered_colors[-1] - colors[i + 1])

        # print("{}. {} Co-linearity:{}".format(i, stop_indices[i], co_linearity-1))
        if co_linearity - 1 > co_linear_threshold:
            # print("\t Adding:{}".format(stop_indices[i]))
            filtered_stops.append(stop_indices[i])
            filtered_colors.append(colors[i])
    filtered_stops.append(stop_indices[-1])
    filtered_colors.append(colors[-1])
    return np.array(filtered_stops, dtype=np.int), np.array(filtered_colors)


def gradient_test():
    m = np.array([
        [1,4,6],
        [5,7,9],
        [10, 12, 13]
    ], dtype=float)
    gy, gx = np.gradient(m)

    print ("Numpy Gy:\n", gy)
    print ("Numpy Gx:\n", gx)

    gx, gy = grad(m, None, normalize=False) # Our sobel filter convolution
    print ("Our Gy:\n", gy)
    print ("Our Gx:\n", gx)


def double_diff_np(line, filter_size):
    line = th.Tensor(line)
    line = th.cat([line[0] * th.ones(filter_size), line, line[-1] * th.ones(filter_size)])
    return tensor2numpy(th.abs(double_diff(line, support=filter_size)[filter_size:len(line) - filter_size]))


def get_peaks(series, size):
    def is_peak(i):
        if i-size < 0 or i+1+size >= len(series):
            return False
        l = 0 if i-size < 0 else i-size
        r = len(series) if i+1+size >= len(series) else i+1+size
        left = np.mean(series[l:i]) < series[i] if l < i else True
        right = series[i] > np.mean(series[i+1:r]) if i < r else True
        return left and right

    candidates = [i for i, s in enumerate(series[1:-1]) if is_peak(i)]
    return candidates


def Matrix(s, theta):
    S=np.array([
        [s, 0.],
        [0., 1.]
    ])
    R=np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    return S@R


def general_radial_color_coord(img, center, ecc, scale, rotation):
    cols_rows = np.array([img.shape[1], img.shape[0]])

    def sample_color(D, s):
        center_img_space = center * cols_rows
        indices = []
        colors = []
        for t in np.arange(0, D, s):
            p = np.round(center_img_space + ecc*t)
            i,j = int(p[1]), int(p[0])

            #FixMe: For masked image, skip this check and rely on D
            if i >= cols_rows[1] or j >= cols_rows[0] or i < 0 or j < 0:
                continue
            if len(indices) == 0 or indices[-1][0] != (i,j):
                indices.append([i,j])
                colors.append(img[i,j])
        colors = np.array(colors)/ 255
        indices = np.array(indices)
        return indices, colors

    indices, colors = sample_color(2*np.linalg.norm(cols_rows), 1)

    if VERBOSE:
        center_img_space = center * cols_rows
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.plot(center_img_space[0], center_img_space[1], 'bo')
        if len(indices) > 0:
            ax.plot(indices[:,1], indices[:,0], '--')

        # x = np.arange(0, len(colors), 1)
        # ax[1].plot(x, colors[:, 0], 'r-')
        # ax[1].plot(x, colors[:, 1], 'g-')
        # ax[1].plot(x, colors[:, 2], 'b-')

        plt.show()

    return colors, indices


def color_profile(img, center_img_space, ecc_img_space, scale, rotation):
    cols_rows = np.array([img.shape[1], img.shape[0]])
    RADIUS = 0 #int(np.max(cols_rows)/16)
    rows, cols = cols_rows[1], cols_rows[0]
    r = np.sqrt(rows**2 + cols**2)
    ecc_mag = np.linalg.norm(ecc_img_space)
    theta = 0 if ecc_mag < ERROR else np.arctan2(ecc_img_space[1] / ecc_mag, ecc_img_space[0] / ecc_mag)
    ex = center_img_space + r * np.array([np.cos(theta), np.sin(theta)])
    print("Center:{}, End:{}".format(center_img_space, ex))

    profile = []
    coordinate = []

    def transform(p):
        return p
        # T = Matrix(scale, rotation)
        # # print(T, p)
        # return p @ T

    def index(p):
        xy = np.round(p)
        return int(xy[1]), int(xy[0])

    great_circles = []

    def avg_arc_color(center, point_on_circumference, draw):
        radius = np.linalg.norm(center-point_on_circumference)
        colors = []
        circle = []
        for t in np.arange(0, np.pi*2, np.pi*2/100):
            p = center + transform(radius * np.array([np.cos(t), np.sin(t)]))
            i,j = index(p)
            if RADIUS <= i < rows-RADIUS and RADIUS <= j < cols-RADIUS:
                colors.append(img[i,j])
                if draw:
                    circle.append([i, j])
        if len(colors) > 0:
            if draw:
                great_circles.append(np.array(circle))
            return np.mean(colors, axis=0)
        else:
            return None

    N = int(r)
    count = 0

    def get_y(px):
        return center_img_space[1] + (ex[1] - center_img_space[1]) * (px - center_img_space[0]) / (ex[0] -center_img_space[0])
    print("Data Items : {}".format(int(ex[0] - center_img_space[0])))
    for px in np.arange(center_img_space[0], ex[0], np.sign(ex[0]-center_img_space[0])):
        py = get_y(px)
        count += 1
        p = np.array([px, py])
        color = avg_arc_color(center_img_space, point_on_circumference=p, draw=(count % int(N/20)==0))
        if color is not None:
            pi, pj = index(p)
            if len(coordinate) == 0 or coordinate[-1] != (pj, pi):
                profile.append(color)
                coordinate.append((pj, pi))
        else:
            break

    print ("Great Arcs:", len(great_circles))
    if VERBOSE:
        plt.imshow(img)
        plt.plot([center_img_space[0], ex[0]], [center_img_space[1], ex[1]], 'o--')
        # for arc in great_circles:
        #     plt.plot(arc[:,1], arc[:,0], '--')
        plt.show()

    return np.array(profile), np.array(coordinate)


def extract_stops(red_ori, green_ori, blue_ori, filter_size):
    filter_size = 10
    red, blue, green = avg_smoothen1d(red_ori, filter_size), avg_smoothen1d(green_ori, filter_size), \
                       avg_smoothen1d(blue_ori, filter_size)
    # red, blue, green = red_ori, green_ori, blue_ori

    r_dd = g_smoothen(double_diff_np(red, filter_size), filter_length=10)
    g_dd = g_smoothen(double_diff_np(green, filter_size), filter_length=10)
    b_dd = g_smoothen(double_diff_np(blue, filter_size), filter_length=10)

    peak_r = get_peaks(r_dd, filter_size)
    peak_g = get_peaks(g_dd, filter_size)
    peak_b = get_peaks(b_dd, filter_size)

    candidates = peak_r + peak_g + peak_b
    candidates += [0] + [len(r_dd)-1]
    candidates = list(set(candidates))
    candidates.sort()

    window_size = max(5, int(len(red)/ 30))
    window_size = 5
    stop_indices = candidates # rolling_window_cluster(candidates, window_size)

    colors = np.array([[red_ori[i], green_ori[i], blue_ori[i]] for i in stop_indices])
    stop_indices, colors = filter_color_stops(stop_indices, colors, co_linear_threshold=0.05)
    stop_indices = rolling_window_cluster(stop_indices, window_size=10)
    colors = np.array([[red_ori[i], green_ori[i], blue_ori[i]] for i in stop_indices])

    if VERBOSE:
        fig, axs = plt.subplots(2, 3)
        N = len(red)
        axs[0,0].plot(np.arange(0, N, 1), red)
        axs[0,1].plot(np.arange(0, N, 1), green)
        axs[0,2].plot(np.arange(0, N, 1), blue)

        axs[1,0].plot(np.arange(0, N, 1), r_dd)
        axs[1,1].plot(np.arange(0, N, 1), g_dd)
        axs[1,2].plot(np.arange(0, N, 1), b_dd)

        axs[1,0].plot(stop_indices, r_dd[stop_indices], 'o', color='red')
        axs[1, 1].plot(stop_indices, g_dd[stop_indices], 'o', color='red')
        axs[1, 2].plot(stop_indices, b_dd[stop_indices], 'o', color='red')
        plt.show()
    return stop_indices, colors


def grad_np(color_img):
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
    return grad_np(img[:, :, 0]), grad_np(img[:, :, 1]), grad_np(img[:, :, 2])


def get_circle(center, radius, scale, rotation, N=100):
    thetas = np.arange(0, np.pi*2+np.pi*2/N, np.pi*2/N)
    points = np.array([
        [radius*np.cos(theta), radius*np.sin(theta)] for theta in thetas
    ])

    scale = 1 if abs(1-scale) < 0.05 else scale
    T = Matrix(1/scale, rotation)
    points = points @ T

    points = points + center[None, :]

    return points


def read_img(name=None):
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

    import os
    from hyperparam import IMAGE_STOCK
    # IMAGE_STOCK = "/Users/souchakr/GIT/GradientExtraction/Image_stock/"
    img = read(os.path.join(IMAGE_STOCK, 'Radial', 'General', 'ecc_r45.png' if name is None else name))
    return img


if __name__ == '__main__':
    get_circle(np.zeros(2), 128, 1/.6, np.pi/4)
    # gradient_test()

