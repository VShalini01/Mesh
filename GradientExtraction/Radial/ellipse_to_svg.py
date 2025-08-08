import numpy as np
import matplotlib.pyplot as plt
from hyperparam import ERROR, VERBOSE
from misc import g_smoothen, avg_smoothen1d, double_diff_np, rolling_window_cluster, filter_color_stops, Matrix
from Draw.svg import create_ellipse


def radius(p, o, e):
    op = (o[None, :] - p)
    a = (e.T @ e - 1)
    b = 2 * (e[None, :] * op).sum(axis=1)
    c = (op * op).sum(axis=1)
    return (-b - np.sqrt(-(4 * a * c) + (b ** 2))) / (2 * a)


def color_profile(img, c, ecc, s, t):
    cols_rows = np.array([img.shape[1], img.shape[0]])
    c = c * cols_rows

    ecc = ecc * cols_rows
    RADIUS = int(np.max(cols_rows)/16)
    rows, cols = cols_rows[1], cols_rows[0]
    r = np.sqrt(rows**2 + cols**2)
    ecc_mag = np.linalg.norm(ecc)
    theta = 0 if ecc_mag < ERROR else np.arctan2(ecc[1]/ecc_mag, ecc[0]/ecc_mag)
    ex = c + r * np.array([np.cos(theta), np.sin(theta)])
    profile = []
    coordinate = []
    for h in np.arange(0, 1, 1/r):
        p = c * (1-h) + ex * h
        p = np.round(p)
        if RADIUS <= p[0] <= cols-RADIUS and RADIUS <= p[1] <= rows-RADIUS:
            px, py = int(p[0]), int(p[1])
            if len(coordinate) == 0 or coordinate[-1] != (px, py):
                profile.append([img[py, px, 0], img[py, px, 1], img[py, px, 2]])
                coordinate.append((px,py))
        else:
            break
    return np.array(profile), np.array(coordinate)


def get_peaks(series, size):
    def is_peak(i):
        if i-size < 0:
            return False
        l = 0 if i-size < 0 else i-size
        r = len(series) if i+1+size >= len(series) else i+1+size
        left = np.mean(series[l:i]) < series[i] if l < i else True
        right = series[i] > np.mean(series[i+1:r]) if i < r else True
        # print("{}:Peak ? {}: L-mean:{}, R-mean:{}".format(i, series[i], np.mean(series[l:i]), np.mean(series[i+1:r])))
        return left and right

    candidates = [i for i, s in enumerate(series[1:]) if is_peak(i)]
    return candidates


def extract_stops(red_ori, green_ori, blue_ori, filter_size):
    red, blue, green = avg_smoothen1d(red_ori, filter_size), avg_smoothen1d(green_ori, filter_size), \
                       avg_smoothen1d(blue_ori, filter_size)
    # red, blue, green = red_ori, green_ori, blue_ori

    r_dd = g_smoothen(double_diff_np(red, filter_size), filter_length=3)
    g_dd = g_smoothen(double_diff_np(green, filter_size), filter_length=3)
    b_dd = g_smoothen(double_diff_np(blue, filter_size), filter_length=3)

    print("Series size:{}. Filter size:{}".format(len(r_dd), filter_size))
    peak_r = get_peaks(r_dd, filter_size)
    peak_g = get_peaks(g_dd, filter_size)
    peak_b = get_peaks(b_dd, filter_size)

    candidates = peak_r + peak_g + peak_b
    candidates += [0]
    candidates = list(set(candidates))
    candidates.sort()

    window_size = max(5, int(len(red)/ 30))
    stop_indecies = rolling_window_cluster(candidates, window_size)
    colors = np.array([[red_ori[i], green_ori[i], blue_ori[i]] for i in stop_indecies])
    stop_indecies, colors = filter_color_stops(stop_indecies, colors, co_linear_threshold=1)

    if VERBOSE:
        fig, axs = plt.subplots(2, 3)
        N = len(red)
        axs[0,0].plot(np.arange(0, N, 1), red)
        axs[0,1].plot(np.arange(0, N, 1), green)
        axs[0,2].plot(np.arange(0, N, 1), blue)

        axs[1,0].plot(np.arange(0, N, 1), r_dd)
        axs[1,1].plot(np.arange(0, N, 1), g_dd)
        axs[1,2].plot(np.arange(0, N, 1), b_dd)

        axs[1,0].plot(stop_indecies, r_dd[stop_indecies], 'o', color='red')
        axs[1, 1].plot(stop_indecies, g_dd[stop_indecies], 'o', color='red')
        axs[1, 2].plot(stop_indecies, b_dd[stop_indecies], 'o', color='red')
        plt.show()
    return stop_indecies, colors


def create_svg(img, c, ecc, scale, theta):
    cols_rows = np.array([img.shape[1], img.shape[0]])
    # Colors and Stops
    from misc import general_radial_color_coord, extract_stops, get_circle
    profile, coordinates = general_radial_color_coord(img, c, ecc, scale, theta) # color_profile(img, c*cols_rows, ecc*cols_rows, s, t)
    filter_size = int(np.round(np.max(cols_rows) * 3 / 32))
    stop_indices, colors = extract_stops(profile[:, 0], profile[:, 1], profile[:, 2], filter_size)

    focal_point = c * cols_rows
    outer_point = np.array([coordinates[-1, 1], coordinates[-1, 0]])
    outer_radius = radius(p=outer_point, o=focal_point, e=ecc).squeeze()
    outer_center = focal_point + outer_radius * ecc

    print("\n--------------------------------------------------------------")
    print(" Transformed space:  As one would see in the plot on the image.")
    print("--------------------------------------------------------------")
    print("\t- Focal Point: {}\n\t- Outer Center: {}\n\t- Outer Radius: {}".format(
        focal_point, outer_center, outer_radius
    ))
    print("------------------------------------------------------------")

    if VERBOSE:
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(img)
        ax[0].plot(coordinates[:,1], coordinates[:,0], '--')
        for stop_index in stop_indices:
            ax[0].plot(coordinates[stop_index, 1], coordinates[stop_index, 0], 'bo')

        x = np.arange(0, len(profile), 1)
        ax[1].plot(x, profile[:,0], 'r-')
        ax[1].plot(x, profile[:, 1], 'g-')
        ax[1].plot(x, profile[:, 2], 'b-')

        for stop_index in stop_indices:
            ax[1].plot([x[stop_index], x[stop_index]], [0, 1], '--', color='gray')

        ax[0].plot(focal_point[0], focal_point[1], '*', color='black')
        ax[0].plot(outer_center[0], outer_center[1], '*', color='black')
        ax[0].plot(outer_point[0], outer_point[1], '*', color='black')

        circle = get_circle(outer_center, outer_radius, scale, theta)
        ax[0].plot(circle[:, 0], circle[:, 1], '--')

        plt.show()




    # create_ellipse(size=(cols_rows[1], cols_rows[0]), stops=stops, colors=colors, focal=focal_iso_img,
    #                outer_center=stops[-1], outer_radius=np.linalg.norm(stops[0]-stops[-1]), rotation= t, scale=s)

