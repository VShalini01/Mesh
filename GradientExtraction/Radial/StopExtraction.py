import numpy as np
import matplotlib.pyplot as plt
from GradientExtraction.Radial.hyperparam import VERBOSE
from GradientExtraction.Radial.misc import laplacian1d, avg_smoothen1d
from GradientExtraction.Radial.helper import radius


def index(point):
    point = np.round(point)
    return int(point[1]), int(point[0])


def get_circular_color_sample(img, mask, c, p):
    N = 100
    samples = []
    r = np.linalg.norm(p-c)
    for t in np.arange(0, 2 * np.pi, 2 * np.pi / N):
        i, j = index(c + r * np.array([np.cos(t), np.sin(t)]))
        if 0 <= i < img.shape[0] and 0 <= j < img.shape[1] and (mask is None or mask[i,j]):
            samples.append(img[i, j])
    if len(samples) == 0:
        return None
    else:
        return np.mean(samples, axis=0)


def get_color_profile(img, center, ecc, mask):
    rows, cols = img.shape[0], img.shape[1]
    print("center", center)
    c_i, c_j = index(center)
    ecc_theta = np.arctan2(ecc[1], ecc[0])

    profile = []
    coords = []

    if VERBOSE:
        circles = []

    for r in range(rows + cols):
        p = np.array([c_j, c_i]) + r * np.array([np.cos(ecc_theta), np.sin(ecc_theta)])
        r_p = radius(p.reshape(1, -1), center, ecc).squeeze()
        center_p = center + r_p*ecc
        color = get_circular_color_sample(img, mask, center_p, p)
        if color is not None:
            profile.append(color)
            coords.append(index(p))

        if VERBOSE and len(profile) % int(max(rows, cols)*40/512) == 0:
            circles.append((center_p, p, r_p))

    if VERBOSE:
        N=100
        l = np.arange(0, 2*np.pi, 2*np.pi/N)
        fig, ax = plt.subplots()
        ax.imshow(img)
        for c, p, r in circles:
            # r = np.linalg.norm(c-p)
            ax.plot(c[0] + r*np.cos(l), c[1] + r*np.sin(l), '-', color='black', alpha=0.1)
        plt.show()
    return np.array(profile), np.array(coords)


def get_peaks(s):
    size = 10

    def lSample(i):
        return s[max(0, i-size):i]

    def rSample(i):
        return s[i+1: min(len(s), i+size)]

    candidates = []
    for i in range(len(s)):
        if i == 0:
            if s[0] > np.mean(rSample(i)):
                candidates.append(i)
        elif i == len(s)-1:
            if s[-1] > np.mean(lSample(i)):
                candidates.append(i)
        elif s[i] > np.mean(lSample(i)) and s[i] > np.mean(rSample(i)):
            candidates.append(i)

    return np.array(candidates)


def cluster(profile, s):
    cluster = []
    window = 1
    filtered_s = []
    for i in range(len(s)):
        if len(cluster) == 0:
            cluster.append(s[i])
        elif cluster[-1] + window >= s[i]:
            cluster.append(s[i])
        else:
            max_i = np.argmax(profile[cluster])
            filtered_s.append(cluster[max_i])
            cluster = [s[i]]
    if len(cluster) >0 :
        max_i = np.argmax(profile[cluster])
        filtered_s.append(cluster[max_i])
    return filtered_s


def co_linearity(profile, s, thresold):
    #FixMe: color distance ratio must also match space distance ratio
    f = [s[0]]

    def non_colinearity(i):
        prev = profile[f[-1]]
        curr = profile[s[i]]
        next = profile[s[i + 1]]
        t = (s[i] - f[-1]) / (s[i+1] - f[-1])
        return np.linalg.norm(curr - ((1-t)*prev + t*next))

    for i in range(1, len(s)-1):
        # prev_color = profile[f[-1]]
        # curr_color = profile[s[i]]
        # next_color = profile[s[i+1]]
        q = non_colinearity(i)
        # print("\t {}: Non co-linearity:{}".format(s[i], q))
        if q > thresold:
            f.append(s[i])
    f.append(s[-1])
    return f


def concentric(img, center, mask=None):
    profile, coords = get_color_profile(img, center, np.zeros(2), mask)
    bumps = [np.abs(laplacian1d(profile[:,i])) for i in range(3)]
    peaks = [get_peaks(bumps[i]) for i in range(3)]

    if VERBOSE:
        fig, axs = plt.subplots(3, 3, figsize=(15, 10))
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
    threshold=0.008*np.max(bumps)
    peaks = co_linearity(profile, peaks, thresold=threshold)
    if VERBOSE: print("Threshold:{}, Peaks:{} ".format(threshold, peaks))

    if VERBOSE:
        x = np.arange(0, len(profile))
        for i in range(3):
            axs[2, i].plot(x, bumps[i])
            axs[2, i].plot(peaks, bumps[i][peaks], 'o', color='red')
        plt.show()

    return profile[peaks], coords[peaks]


def eccentric(img, center, ecc, mask=None):
    profile, coords = get_color_profile(img, center, ecc=ecc, mask=mask)
    bumps = [np.abs(laplacian1d(profile[:,i])) for i in range(3)]
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
        plt.show()

    return profile[peaks], coords[peaks]


def stop_percentage(center_coord, stop_coords):
    lenghts = [np.linalg.norm(center_coord-stop_coords[0])] + [np.linalg.norm(stop_coords[i]-stop_coords[i+1])
                                                               for i in range(len(stop_coords)-1)]
    lenghts = np.cumsum(lenghts)
    return lenghts/lenghts[-1]


def salient_color_profile(coords, profile, debug):

    bumps = [np.abs(avg_smoothen1d(profile[:, i], size=20)) for i in range(3)]
    peaks = [get_peaks(bumps[i]) for i in range(3)]

    if debug:
        fig, axs = plt.subplots(3, 3)
        x = np.arange(0, len(profile))
        for i in range(3):
            axs[0, i].plot(x, profile[:, i])
            axs[1, i].plot(x, bumps[i])
            peak = np.array(peaks[i]).astype(int)
            f_peak = cluster(bumps[i], peak)
            axs[1, i].plot(peak, bumps[i][peak], 'o', color='red')
            axs[1, i].plot(f_peak, bumps[i][f_peak], 'o', color='blue')

    peaks = np.sort(list(set(cluster(bumps[0], peaks[0]) + cluster(bumps[1], peaks[1]) + cluster(bumps[2], peaks[2]))))
    if len(peaks) < 1: return None
    if debug: print("Peaks ", peaks)
    threshold = 0.5 * np.max(bumps)
    peaks = co_linearity(profile, peaks, thresold=threshold)
    if debug: print("Threshold:{}, Peaks:{} ".format(threshold, peaks))

    if debug:
        x = np.arange(0, len(profile))
        for i in range(3):
            axs[2, i].plot(x, bumps[i])
            axs[2, i].plot(peaks, bumps[i][peaks], 'o', color='red')
        plt.show()

    f_coords = [coords[0]]
    f_colors = [profile[0]]
    for p in peaks:
        f_coords.append(coords[p])
        f_colors.append(profile[p])
    f_coords.append(coords[-1])
    f_colors.append(profile[-1])

    return np.array(f_coords), np.array(f_colors)