import numpy as np
import torch as th
from misc import grad, smoothen, rolling_window_cluster, tensor_img, g_smoothen, laplacian1d
from hyperparam import VERBOSE
from Draw.draw import vector_field, sanitize
import matplotlib.pyplot as plt


RADIUS = 10


def color_profile(img, c1, c2):
    rows, cols = img.shape[1], img.shape[2]
    r = np.sqrt(rows**2 + cols**2)
    theta = np.arctan2((c2-c1)[1], (c2-c1)[0])
    ex = c1 + r * np.array([np.cos(theta), np.sin(theta)])
    profile = []
    coordinate = []
    for h in np.arange(0, 1, 1/r):
        p = c1 * (1-h) + ex * h
        if RADIUS <= p[0] <= cols-RADIUS and RADIUS <= p[1] <= rows-RADIUS:
            px, py = int(p[0]), int(p[1])
            if len(coordinate) == 0 or coordinate[-1] != (px, py):
                profile.append([img[0, py, px], img[1, py, px], img[2, py, px]])
                coordinate.append((px,py))
        else:
            break
    return np.array(profile), np.array(coordinate)


def extract_stops(profile, coordinates, window_size):
    smoother_profile0 = g_smoothen(profile[:, 0])
    smoother_profile1 = g_smoothen(profile[:, 1])
    smoother_profile2 = g_smoothen(profile[:, 2])

    la_profile0 = np.abs(laplacian1d(smoother_profile0))
    la_profile1 = np.abs(laplacian1d(smoother_profile1))
    la_profile2 = np.abs(laplacian1d(smoother_profile2))

    threshold = [2*np.mean(la_profile0), 2*np.mean(la_profile1), 2*np.mean(la_profile2)]
    candidates = []
    for i,c in enumerate(coordinates):
        if la_profile0[i] >= threshold[0]:
            candidates.append(i)
        elif la_profile1[i] >= threshold[1]:
            candidates.append(i)
        elif la_profile2[i] >= threshold[2]:
            candidates.append(i)

    candidates.append(len(candidates)-1)

    stop_indices = rolling_window_cluster(candidates, window_size)
    colors = np.array([profile[i] for i in stop_indices])
    # stop_indices, colors = filter_color_stops(stop_indices, colors)

    if VERBOSE:
        fig, axs = plt.subplots(2, 3)
        length = len(profile)
        # axs[0, 0].plot(np.arange(0, length), profile[:, 0])
        # axs[0, 1].plot(np.arange(0, length), profile[:, 1])
        # axs[0, 2].plot(np.arange(0, length), profile[:, 2])

        axs[0, 0].plot(np.arange(0, length), smoother_profile0)
        axs[0, 1].plot(np.arange(0, length), smoother_profile1)
        axs[0, 2].plot(np.arange(0, length), smoother_profile2)

        axs[1, 0].plot(np.arange(0, length), la_profile0)
        axs[1, 0].plot(np.arange(0, length), threshold[0]*np.ones(length))
        axs[1, 1].plot(np.arange(0, length), la_profile1)
        axs[1, 1].plot(np.arange(0, length), threshold[1]*np.ones(length))
        axs[1, 2].plot(np.arange(0, length), la_profile2)
        axs[1, 2].plot(np.arange(0, length), threshold[2]*np.ones(length))
        plt.show()
    stops = np.array([coordinates[s] for s in stop_indices])
    return stops, colors


def statistical_method(img,Gs):
    from Radial.ring import estimate
    inner_center, inner_radius, outer_center, outer_radius = estimate(img, Gs)

    axis = (outer_center - inner_center) / np.linalg.norm(inner_center - outer_center)
    focal_point = inner_center - 0.5 * inner_radius * axis
    outer_stop = outer_center + outer_radius * axis

    if VERBOSE:
        plt.imshow(sanitize(img))
        plt.plot(focal_point[0], focal_point[1], 'o')
        plt.plot([focal_point[0], outer_stop[0]], [focal_point[1], outer_stop[1]], '--', color='black')
        plt.text(focal_point[0], focal_point[1], 'f')
        plt.plot(inner_center[0], inner_center[1], '+')
        plt.plot(outer_stop[0], outer_stop[1], '+')
        plt.plot(outer_stop[0], outer_stop[1], 'o')

        plt.plot(outer_center[0], outer_center[1], '+')
        plt.text(outer_center[0], outer_center[1], 'c')
        plt.show()

    profile, coordinates = color_profile(img, focal_point, outer_stop)
    window_size = max(10, int(max(img.shape[1], img.shape[2]) / 20))
    stops, colors = extract_stops(profile, coordinates, window_size)

    for s, c in zip(stops, colors):
        print("@ {} -> {}".format(s, c))

    if VERBOSE:
        plt.imshow(sanitize(img))
        plt.plot(stops[:, 0], stops[:, 1], '--o')
        plt.plot(focal_point[0], focal_point[1], '*', color='black')
        plt.plot(outer_center[0], outer_center[1], '*', color='black')
        plt.plot(outer_stop[0], outer_stop[1], 'o', color='black')
        plt.show()

    return stops, colors, focal_point, outer_center, outer_radius


def analytic_method(img, Gs):
    from Radial.mikes_method import center_eccentricity
    f, e, res = center_eccentricity(Gs)
    eccentricity = np.linalg.norm(e)
    if VERBOSE:
        print("eccentricity:{}, center:{}".format(eccentricity, f))
        plt.imshow(sanitize(img))
        plt.plot(f[0], f[1], 'o', color='black')
        l = 100
        plt.plot([f[0], f[0] + l*e[0]], [f[1], f[1]+l*e[1]], '-')
        plt.show()

    profile, coordinates = color_profile(img, f, f+e)
    window_size = max(10, int(max(img.shape[1], img.shape[2]) / 20))
    stops, colors = extract_stops(profile, coordinates, window_size)

    def radius(p):
        c = -(p-f).T@(p-f)
        a = 1 - e.T@e
        b = 2 * (p-f).T@e
        return (-b + np.sqrt(b**2 - 4*a*c))/(2*a)

    r = radius(stops[-1])
    c = stops[-1] - e*r/np.linalg.norm(e)

    if VERBOSE:
        print("Outer Radius:", r)
        plt.imshow(sanitize(img))
        plt.plot(stops[:, 0], stops[:, 1], '--o')
        plt.plot(f[0], f[1], '*', color='black')
        plt.plot(c[0], c[1], '*', color='black')
        l = np.arange(0,2*np.pi, np.pi/50)
        plt.plot(c[0] + r*np.cos(l), c[1] + r*np.sin(l), '--')
        plt.show()
    return stops, colors, f, c, r


def ellipses(img, Gs):
    from Radial.ellipse import foci
    o, e, a = foci(Gs)
    print("Origin:", o, "Eccentricity:", e, "Scale:",a)
    c = o*a
    c2 = o + 100*e*a
    plt.imshow(sanitize(img))
    plt.plot(c[0], c[1], 'o')
    plt.plot([c[0], c2[0]], [c[1], c2[1]], '-')
    # plt.plot([c1[0], c2[0]], [c1[1], c2[1]], 'o-')
    plt.show()

    assert 0


def reconstruct(img):
    if img.shape[-1] == 4:
        mask = th.Tensor((img[:, :, 3] > 100).astype(np.int))
    else:
        mask = th.ones(size=[img.shape[0], img.shape[1]])

    img = img[:, :, :3]
    img = smoothen(tensor_img(img))

    Gr, Gg, Gb = grad(img[0], mask, normalize=True), grad(img[1], mask, normalize=True), grad(img[2], mask, normalize=True)
    if VERBOSE:
        vector_field(img, mask, Gr[0], Gr[1], "Red Gradient Field")
        vector_field(img, mask, Gg[0], Gg[1], "Green Gradient Field")
        vector_field(img, mask, Gb[0], Gb[1], "Blue Gradient Field")
    return analytic_method(img, [Gr, Gg, Gb])


