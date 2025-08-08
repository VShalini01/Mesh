import numpy as np
import matplotlib.pyplot as plt
import config as cg
from StopExtraction import salient_color_profile


def pixels2img(pxs, ori_image):
    C = ori_image.shape[2]
    l, r = np.min(pxs[:,1]), np.max(pxs[:,1])
    b, t = np.min(pxs[:, 0]), np.max(pxs[:, 0])
    one = 255 if np.max(ori_image) > 2 else 1
    offset = np.array([l, b])
    img = np.zeros(shape=(t-b+1, r-l+1, C+1))
    for i, j in pxs:
        img[i-b, j-l, :C] = ori_image[i, j, :C]
        img[i-b, j-l, C] = one
    img = img.astype(np.uint8)
    return img, offset


def get_sample(image, theta, debug):
    H, W = image.shape[:2]
    def point(s, d, t):
        return s + d*np.array([np.cos(t), np.sin(t)])

    def line_seg(s, t, l, r):
        return np.array([point(s, d, t) for d in np.arange(l, r)])

    length = np.linalg.norm(image.shape[:2])
    com = np.array([image.shape[1], image.shape[0]])*0.5
    centers = line_seg(com, theta, -length, length)
    coords, profile = [], []
    for c in centers:
        ps = np.round(line_seg(c, theta + np.pi/2, -length, length))
        cs = [
            image[int(p[1]), int(p[0]), : 3] for p in ps
              if 0 <= p[1] < H and 0 <= p[0] < W and image[int(p[1]), int(p[0]), 3] > 0
        ]
        if len(cs) > cg.SIGNIFICANT_COLOR_SAMPLE :
            profile.append(np.std(cs, axis=0))
            coords.append(c)

    if len(coords) < 2: return None, None

    coords, profile = np.array(coords), np.array(profile)
    f_coords_colors = salient_color_profile(coords=coords, profile=profile, debug=debug)

    if debug and f_coords_colors is not None:
        f_coords, _ = f_coords_colors
        plt.imshow(image)
        plt.plot(f_coords[:, 0], f_coords[:, 1], 'o-')
        plt.show()
    return f_coords_colors

