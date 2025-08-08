import numpy as np
import torch as th
from torch import nn
from misc import grad, tensor2numpy, double_diff, smoothen, tensor_img, rotate_with_mask, rolling_window_cluster, rotate_point, g_smoothen
from hyperparam import VERBOSE
from progress.bar import ChargingBar
import statistics
import matplotlib.pyplot as plt
from Draw.draw import vector_field, image_direction, sanitize
from misc import rotate_without_pad, filter_color_stops, get_peaks
from copy import deepcopy
from scipy.optimize import least_squares
from tools import timer


def get_line(shape, theta):
    # FixMe: This only works for rectangular patches
    from skimage.draw import line
    radius = np.linalg.norm(shape) * 0.5

    start = int(-radius * np.cos(theta)), int(-radius * np.sin(theta))
    end = int(radius * np.cos(theta)), int(radius * np.sin(theta))

    coordinates = np.array(list(zip(*line(*start, *end))))
    coordinates = np.array([[
        -p[1]+shape[0]*0.5, p[0]+shape[1]*0.5] for p in coordinates
        if 0 <= -p[1]+shape[0]*0.5 < shape[0] and 0 <= p[0]+shape[1]*0.5 < shape[1]
    ])
    return coordinates


def linear_blend(mask, stop_indices, colors):
    assert colors.shape[-1] == 3
    rows, cols = mask.shape[0], mask.shape[1]
    render = np.zeros(shape=(rows, cols, 4))
    for c in range(3):
        for k, stop in enumerate(stop_indices):
            if k == 0:
                render[:, :stop, c] = colors[k,c]*np.ones(shape=(rows, stop))

            if k == len(stop_indices)-1:
                render[:, stop:, c] = colors[k, c]*np.ones(shape=(rows, cols-stop))
            else:
                assert stop_indices[k+1] - stop > 1
                r = 1/(stop_indices[k+1]-stop)
                chunk = np.tile(np.arange(0, 1, r), (render.shape[0],1))
                chunk_c = colors[k,c] * (1-chunk) + colors[k+1,c] * chunk
                # print("{} - {}, r:{:.3f}, chunk Shape :{}, render shape :{}"
                #       "".format(stop, stop_indices[k+1], r, chunk_c.shape, render[:, stop:stop_indices[k+1], c].shape))
                render[:, stop: stop+chunk_c.shape[1], c] = chunk_c
    render[:, :, 3] = mask
    return render


def linear_blend_refine(img, mask, stop_indices, colors):
    assert colors.shape[-1] == 3
    assert img.shape[:2] == mask.shape
    if img.shape[2] == 4:
        img = img[:,:,:3]
    rows, cols = mask.shape[0], mask.shape[1]
    x0 = colors.flatten()

    def fun(x):
        clr = x.reshape(-1, 3)
        render = np.zeros(shape=(rows, cols, 3))
        for c in range(3):
            for k, stop in enumerate(stop_indices):
                if k == 0:
                    render[:, :stop, c] = clr[k, c] * np.ones(shape=(rows, stop))

                if k == len(stop_indices) - 1:
                    render[:, stop:, c] = clr[k, c] * np.ones(shape=(rows, cols - stop))
                else:
                    assert stop_indices[k + 1] - stop > 1
                    r = 1 / (stop_indices[k + 1] - stop)
                    chunk = np.tile(np.arange(0, 1, r), (render.shape[0], 1))
                    chunk_c = clr[k, c] * (1 - chunk) + clr[k + 1, c] * chunk
                    render[:, stop: stop + chunk_c.shape[1], c] = chunk_c
        return np.sum(mask * np.linalg.norm(render-img, axis=2))

    res = least_squares(fun, x0)
    new_colors = res.x.reshape(-1, 3)

    return linear_blend(mask, stop_indices, new_colors)

@timer
def color_along(img, theta, mask):
    """ Rotate the image to align the gadient along x-axis. Estimate gradient stops and colors in this space.
    Return the rotated stop location and the reconstructed image.
    """
    r_img, pad = rotate_with_mask(img, mask, -theta)

    red_img = r_img[:, :, 0]
    green_img = r_img[:, :, 1]
    blue_img = r_img[:, :, 2]
    mask = r_img[:, :, 3]
    mask = (mask>0).astype(np.float)

    sum_mask = np.sum(mask, axis=0)
    sum_red = np.sum(red_img*mask, axis=0)
    sum_green = np.sum(green_img*mask, axis=0)
    sum_blue = np.sum(blue_img*mask, axis=0)

    assert sum_mask.shape == sum_red.shape == sum_green.shape == sum_blue.shape and sum_blue.shape and sum_blue.shape[0] == red_img.shape[1]

    red, green, blue = [], [], []
    for i, m in enumerate(sum_mask):
        if m > 0 and sum_red[i] <= m:
            red.append((i, sum_red[i]/m))
        if m > 0 and sum_green[i] <= m:
            green.append((i, sum_green[i]/m))
        if m > 0 and sum_blue[i] <= m:
            blue.append((i, sum_blue[i]/m))

    red, green, blue = np.array(red), np.array(green), np.array(blue)

    filter_support = int(10 * np.max(img.shape[:2]) / 128)

    def dd(line):
        line = th.Tensor(line)
        line = th.cat([line[0]*th.ones(filter_support), line, line[-1]*th.ones(filter_support)])
        return tensor2numpy(th.abs(double_diff(line, support=filter_support)[filter_support:len(line) - filter_support]))

    dd_red, dd_green, dd_blue = g_smoothen(dd(red[:,1])), g_smoothen(dd(green[:,1])), g_smoothen(dd(blue[:,1]))

    filter_support = 10
    peak_r = get_peaks(dd_red, filter_support)
    peak_g = get_peaks(dd_green, filter_support)
    peak_b = get_peaks(dd_blue, filter_support)

    if VERBOSE:
        fig, axs = plt.subplots(1, 3)
        axs[0].plot(dd_red)
        axs[0].plot(peak_r, [dd_red[i] for i in peak_r], 'o')
        axs[0].set_title("Red")
        axs[1].plot(dd_green)
        axs[1].plot(peak_g, [dd_green[i] for i in peak_g], 'o')
        axs[1].set_title("Green")
        axs[2].plot(dd_blue)
        axs[2].plot(peak_b, [dd_blue[i] for i in peak_b], 'o')
        axs[2].set_title("Blue")
        plt.show()


    candidates = [red[i,0] for i in peak_r] + [green[i,0] for i in peak_g] + [blue[i,0] for i in peak_b]

    offset = filter_support
    min_index = np.min([np.min(red[offset:-offset,0]), np.min(green[offset:-offset,0]),
                        np.min(blue[offset:-offset,0])])
    max_index = np.max([np.max(red[offset:-offset,0]), np.max(green[offset:-offset,0]),
                        np.max(blue[offset:-offset,0])])
    candidates.insert(0, min_index), candidates.append(max_index)

    candidates = np.sort(list(set(candidates)))
    window_size = max(10, int(max(img.shape[1], img.shape[2]) / 30))

    stop_indecies = rolling_window_cluster(candidates, window_size)

    colors = []
    r_dict, g_dict, b_dict = dict(red), dict(green), dict(blue)
    for i in stop_indecies:
        colors.append([
            r_dict[i] if i in r_dict else 0, g_dict[i] if i in g_dict else 0, b_dict[i] if i in b_dict else 0
        ])
    colors = np.array(colors)
    assert len(stop_indecies) == len(colors)

    stop_indecies, colors = filter_color_stops(stop_indecies, colors)

    if VERBOSE:
        print("Window size", window_size)
        fig, axs = plt.subplots(3,3)
        axs[0,0].plot(red[:,0], red[:,1])
        axs[0,1].plot(green[:,0], green[:,1])
        axs[0,2].plot(blue[:,0], blue[:,1])
        axs[0, 0].set_title("Red")

        axs[1,0].plot(red[:,0], dd_red)
        axs[1,1].plot(green[:,0], dd_green)
        axs[0,1].set_title("Green")

        axs[1,2].plot(blue[:,0], dd_blue)
        axs[0,2].set_title("Blue")

        axs[2,0].imshow(red_img, cmap='gray', alpha=mask)
        axs[2,1].imshow(green_img, cmap='gray', alpha=mask)
        axs[2,2].imshow(blue_img, cmap='gray', alpha=mask)
        plt.show()

    return img, r_img, pad, mask, stop_indecies, colors, theta


def render(img, r_img, pad, mask, stop_indecies, colors, theta):
    render = linear_blend(mask, stop_indecies, colors) # linear_blend_refine(r_img, mask, stop_indecies, colors)
    render = rotate_without_pad(render, theta)
    shift = np.array([pad[1][0], pad[0][0]])

    # rotate back
    origin =np.array([int(r_img.shape[1]/2), int(r_img.shape[0]/2)])
    stops = np.array([rotate_point(np.array([stop, int(r_img.shape[0]/2)]), -theta, origin) for stop in stop_indecies])

    if VERBOSE:

        plt.imshow(r_img)
        plt.plot(stop_indecies, int(r_img.shape[0]/2)*np.ones(len(stop_indecies)), 'o-')
        plt.title("Stops in Axis Aligned space")
        plt.show()

    # Shift to remove rotation padding.
    stops = stops - shift
    ren_img = render[shift[1]:-shift[1], shift[0]:-shift[0]]
    if VERBOSE:
        fig, axs = plt.subplots(1, 2)
        axs[0].set_title("Original Image")
        axs[0].imshow(sanitize(img))
        axs[1].set_title("Gradient Reconstructed")
        axs[1].imshow(ren_img)
        axs[1].plot(stops[:, 0], stops[:, 1], 'o--')
        plt.show()

    return np.flip(stops, axis=1), colors, ren_img


def get_stops_colors_render(img, theta, mask):
    img, r_img, pad, mask, stop_indecies, colors, theta = color_along(img, theta, mask)
    print("Stop Indecies:", stop_indecies)
    return render(img, r_img, pad, mask, stop_indecies, colors, theta)


def axis(img_c, coordinates):
    assert len(img_c.shape) == 2
    value = th.Tensor([img_c[int(p[0]), int(p[1])] for p in coordinates])
    return value


def solve(Gr, Gg, Gb):
    A = Gr[1]*Gr[0] + Gg[1]*Gg[0] + Gb[1]*Gb[0]
    B = (Gr[1]**2 - Gr[0]**2) + (Gg[1]**2 - Gg[0]**2) + (Gb[1]**2 - Gb[0]**2)

    a, b = th.sum(A), th.sum(B)
    if a == 0:
        print("No solution exits. Probably solid fill.")
        return None
    else:
        return th.arctan((b + th.sqrt(b**2 + 4*a**2)) / (2*a))


def iterative(Gr, Gg, Gb, epoch):
    vec = nn.Parameter(th.Tensor([np.pi]), requires_grad=True)
    # optimizer = th.optim.Adam([vec], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer = th.optim.LBFGS([vec], history_size=10, max_iter=4, line_search_fn="strong_wolfe")
    def objective(theta):
        return th.sum(
                (Gr[0] * th.cos(theta) + Gr[1] * th.sin(theta)) ** 2 +
                (Gg[0] * th.cos(theta) + Gg[1] * th.sin(theta)) ** 2 +
                (Gb[0] * th.cos(theta) + Gb[1] * th.sin(theta)) ** 2
            )

    with ChargingBar('Searching Linear gradient :', max=epoch) as bar:
        for e in range(epoch):
            optimizer.zero_grad()
            en = objective(vec)
            en.backward()
            optimizer.step(lambda: objective(vec))

            bar.bar_suffix = " Energy := {:.4f} | ".format(en.item())
            bar.next()

    return tensor2numpy(vec.detach())[0]

@timer
def direction(img, mask, debug=True):
    assert len(img.shape) == 3 and img.shape[0] == 3

    row, col = img.shape[1], img.shape[2]
    Gr, Gg, Gb = grad(img[0], mask), grad(img[1], mask), grad(img[2], mask)
    Gr = Gr[0][1:row - 1, 1:col - 1], Gr[1][1:row - 1, 1:col - 1]
    Gg = Gg[0][1:row - 1, 1:col - 1], Gg[1][1:row - 1, 1:col - 1]
    Gb = Gb[0][1:row - 1, 1:col - 1], Gb[1][1:row - 1, 1:col - 1]

    if VERBOSE and debug:
        vector_field(img[0], mask, Gr[0], Gr[1], "Red Gradient Field")
        vector_field(img[1], mask, Gg[0], Gg[1], "Green Gradient Field")
        vector_field(img[2], mask, Gb[0], Gb[1], "Blue Gradient Field")

    s_theta = solve(Gr, Gg, Gb)
    return s_theta


def grad_image(size, stops, colors):
    assert len(stops) == len(colors)
    ts = [np.linalg.norm(stops[0]-stop) for stop in stops]
    img = np.zeros(shape=[size[0],size[1],3])
    ray = (stops[1] - stops[0]) / ts[1]

    with ChargingBar('  Reconstructing Gradient :', max=size[0]*size[1]) as bar:
        for i,j in np.ndindex(size):
            x = np.array([i,j])
            d = np.sum(ray*(x-stops[0]))
            if d < 0:
                img[i,j] = colors[0]
            elif d > ts[-1]:
                img[i,j] = colors[len(stops) - 1]
            else:
                for k in range(len(stops)-1):
                    if ts[k] <= d <= ts[k+1]:
                        r = (d-ts[k])/ (ts[k+1] - ts[k])
                        img[i, j] = (1-r)*colors[k] + r*colors[k+1]
                        break
            bar.next()
    return img.astype(np.uint8)


def l2_loss(img, ren_img, original_mask):
    def apply_mask(array):
        return np.stack([array[:, :, 0]*original_mask, array[:, :, 1]*original_mask, array[:, :, 2]*original_mask], axis=2)

    original = (255 * apply_mask(img[:, :, :3])).astype(np.int)
    ren_img = (255 * apply_mask(ren_img[:, :, :3])).astype(np.int)

    mask = (255*np.clip(original_mask, 0, 1)).astype(np.int)

    original = np.stack([original[:,:, 0], original[:,:, 1], original[:,:, 2], mask], axis=2)
    ren_img = np.stack([ren_img[:,:, 0], ren_img[:,:, 1], ren_img[:,:, 2], mask], axis=2)

    diff_img = np.abs(original - ren_img)
    diff_img[:,:,3] = mask
    l2 = np.linalg.norm(diff_img) / (np.sum(original_mask))

    print("Average L2 loss of reconstruction :{:.5f}".format(l2))

    if VERBOSE:
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(original)
        axs[0].set_title("Original")
        axs[1].imshow(ren_img)
        axs[1].set_title("Reconstructed")
        axs[2].imshow(diff_img)
        axs[2].set_title("Difference")
        plt.show()


def get_direction(ori_img, debug):
    """ img is in RGBA/RGB format """
    try:
        if ori_img.shape[-1] == 4:
            mask = th.Tensor((ori_img[:, :, 3] > 100).astype(np.int))
        else:
            mask = th.ones(size=[ori_img.shape[0], ori_img.shape[1]])

        img = deepcopy(ori_img[:, :, :3])
        assert len(img.shape) == 3 and img.shape[2] == 3
        img = smoothen(tensor_img(img))
        theta = -tensor2numpy(direction(img, mask, debug))
        if debug and theta is not None:
            image_direction(sanitize(img), theta, "Linear Gradient direction")

        return theta
    except:
        return None

def reconstruct(ori_img):
    """ img is in RGBA/RGB format """
    if ori_img.shape[-1] == 4:
        mask = th.Tensor((ori_img[:, :, 3] > 100).astype(np.int))
    else:
        mask = th.ones(size=[ori_img.shape[0], ori_img.shape[1]])

    img = deepcopy(ori_img[:, :, :3])
    assert len(img.shape) == 3 and img.shape[2] == 3
    img = smoothen(tensor_img(img))
    theta = tensor2numpy(direction(img, mask))

    if VERBOSE:
        image_direction(sanitize(img), theta, "Linear Gradient direction")

    stops, colors, ren_img = get_stops_colors_render(img, theta, mask)

    # l2_loss(sanitize(img), ren_img, tensor2numpy(mask))

    return stops, (255*colors).astype(np.uint8), sanitize(img)

