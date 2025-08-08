import numpy as np
import matplotlib.pyplot as plt
from hyperparam import IMAGE_STOCK
import cv2
import os
import linear as ln
import Common as cm
import config as cg


SEG_PATH = os.path.join(IMAGE_STOCK, 'Segments')


def read(path):
    img = cv2.imread(path)
    if img is None:
        assert 0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def fit(img, partitions):
    segs = dict()

    def key(c):
        c = np.round(c, 3)
        return (int(c[0]), int(c[1]), int(c[2]))

    for i,j in np.ndindex(img.shape[:2]):
        k = key(partitions[i, j])
        if k not in segs: segs[k] = []
        segs[k].append([i, j])
    return segs


def draw(img, segments, debug):
    plt.imshow(img)
    for seg in segments:
        img_coords_colors = get_stops(context=(img, segments), seg=seg, debug=False)
        if img_coords_colors is not None:
            _, coords, colors = img_coords_colors
            plt.plot(coords[:, 0], coords[:, 1], 'o-', color='red', markersize=2, linewidth=1)

    plt.show()


def get_stops(context, seg, debug):
    ori_image, segments = context
    pxs = segments[seg]
    if len(pxs) <= cg.SIGNIFICANT_COLOR_SAMPLE: return None
    seg_img, offset = cm.pixels2img(np.array(pxs), ori_image)
    theta = ln.get_direction(seg_img, debug=False)
    if theta is not None:
        f_coords_colors = cm.get_sample(seg_img, theta, debug=debug)
        if f_coords_colors is None or f_coords_colors[0] is None:
            return None
        f_coords, f_colors = f_coords_colors
        f_coords = f_coords + offset
        return seg_img, f_coords, f_colors
    else:
        return None


if __name__ == '__main__':
    partition_path = os.path.join(SEG_PATH, "partition.png")
    img_path = os.path.join(SEG_PATH, "pwFO.png")
    part_img = read(partition_path)
    img = read(img_path)
    segs = fit(img, part_img)
    draw(img, segs, debug=True)