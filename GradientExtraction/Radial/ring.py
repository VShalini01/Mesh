import matplotlib.pyplot as plt
import numpy as np
from misc import exterior_angle, sanitize, closest_point, distance
from hyperparam import VERBOSE
from tools import timer

DELTA = 20
RADIUS = 10
SAMPLES = 70
GAMMA = 100 # Number of simulation

def cone(img, pxy, G, delta, radius):
    """Takes a delta step in the tangent direction and retrives a point with same
    color intensity in around the point
    """
    assert delta - radius > 1
    px, py = pxy

    t = np.array([G[1][py, px], G[0][py, px]])

    nx, ny = int(px + delta * t[0]), int(py + delta * t[1])
    ball = (int(2 * radius), int(2 * radius))
    cone_points = dict()

    def direction(x, y):
        p, n, b = np.array([px, py]), np.array([nx, ny]), np.array([x, y])
        return np.linalg.norm(p - b) >= np.linalg.norm(p - n)

    for i, j in np.ndindex(ball):
        x, y = int(i - radius + nx), int(j - radius + ny)

        if radius <= x < img.shape[2] - radius and radius <= y < img.shape[1] - radius and direction(x, y):
            color_diff = np.linalg.norm(img[:, py, px] - img[:, y, x])
            cone_points[(x, y)] = color_diff
        else:  # reached the edge. Information of gradient is invalid
            break

    if len(cone_points) > 0:
        return sorted(cone_points, key=lambda x: cone_points[x])[0]
    else:
        return None


def get_ring(s, G, img):
    rows, cols = img.shape[1], img.shape[2]
    px, py = int(s[0]), int(s[1])
    if G[0][py, px] == 0 and G[1][py, px] == 0:
        return None, None

    ring_pixels = [(px, py)]
    sweep_angle = 0
    for i in range(SAMPLES):
        pxy = ring_pixels[-1]
        nxy = cone(img, pxy, G, DELTA, RADIUS)
        if nxy is not None:
            ring_pixels.append(nxy)
        else:
            break

        if len(ring_pixels) > 2:
            p0, p1, p2 = ring_pixels[-3], ring_pixels[-2], ring_pixels[-1]
            sweep_angle += exterior_angle(p0, p1, p2)
        if sweep_angle >= 360:
            break

    ring_pixels = np.array(ring_pixels)

    if len(ring_pixels) <= 2:
        return None, None
    else:
        return ring_pixels, sweep_angle


def center(G, p1, p2):
    g1x, g1y = G[0][int(p1[1]), int(p1[0])], -G[1][int(p1[1]), int(p1[0])]
    g2x, g2y = G[0][int(p2[1]), int(p2[0])], -G[1][int(p2[1]), int(p2[0])]

    T = np.array([
        [g1x, -g2x],
        [g1y, -g2y]
    ])

    B = np.array([
        [p2[0] - p1[0]],
        [p2[1] - p1[1]]
    ])
    if np.linalg.det(T) == 0:
        return None
    else:
        C = np.linalg.inv(T) @ B
        c1 = np.array([p1[0] + C[0, 0] * g1x, p1[1] + C[0, 0] * g1y])
        return c1


def center_by_intersection(ring_points, G):
    centers = []
    count = len(ring_points)
    assert count > 2
    half = int(count / 2)

    for i in range(half):
        c = center(G, ring_points[half + i], ring_points[i])
        if c is not None:
            centers.append(c)

    if len(centers) > 0:
        return np.mean(centers, axis=0)
    else:
        return None


def get_center(ring, G):
    r_points, sweep = ring
    if sweep > 355:
        c = np.mean(r_points, axis=0)
    else:
        c = center_by_intersection(r_points, G)
    return c


def get_axis(img, G, s):
    rows, cols = img.shape[1], img.shape[2]
    r_points, sweep = get_ring(s, G, img)
    if r_points is None:
        print("Ring not created1")
        return None
    c = get_center((r_points, sweep), G)

    #

    def get_other_ring(c):
        if c is None:
            print("Starting center is None")
            return None, None, None
        s1 = 2 * s - c
        if not (RADIUS < s1[0] <= cols - RADIUS and RADIUS < s1[1] <= rows - RADIUS):
            s1 = 1.5 * s - 0.5 * c
        r_points1, sweep1 = get_ring(s1, G, img)
        direction = 0
        if r_points1 is None:
            print("Outter Ring not created")
            s2 = (s + c) / 2
            r_points1, sweep1 = get_ring(s2, G, img)
            direction = 1

        if r_points1 is not None:
            c1 = get_center((r_points1, sweep1), G)
            if c1 is not None:
                return (c1, c) if direction == 1 else (c, c1), r_points1, sweep1
            else:
                print("Second center could not be estimated")
        return None, None, None

    axis, r_points1, sweep1 = get_other_ring(c)

    if VERBOSE:
        plt.imshow(sanitize(img))
        plt.plot(r_points[:, 0], r_points[:, 1], 'o', markersize=1, color='black')
        if c is not None:
            plt.plot(c[0], c[1], '*', color='black')
        plt.plot(s[0], s[1], 'o', color='black')

        if r_points1 is not None:
            plt.plot(r_points1[:, 0], r_points1[:, 1], 'o', markersize=1, color='black')
        if axis is not None:
            plt.plot(axis[0][0], axis[0][1], '*')
            plt.plot(axis[1][0], axis[1][1], '*')

        plt.show()

    return axis


def find_center(img, G, s):
    rows, cols = img.shape[1], img.shape[2]
    r_points, sweep = get_ring(s, G, img)
    if r_points is not None and sweep >= 45:
        c = get_center((r_points, sweep), G)
        if c is not None and 0 <= c[0] < cols and 0 <= c[1] < rows:
            if VERBOSE and False:
                print("Sweep", sweep, " Ring size ", len(r_points))
                plt.imshow(sanitize(img))
                plt.plot(r_points[:, 0], r_points[:, 1], '--')
                plt.plot(c[0], c[1], 'o', color='black')
                plt.plot(s[0], s[1], '+', color='black')
                plt.show()
            return c, np.mean(np.linalg.norm(r_points - c, axis=1))
    return None, None


def ransac(centers, rows, cols):
    from sklearn import linear_model
    X, Y = centers[:, 0], centers[:, 1]
    X = X.reshape(1, -1).T
    model = linear_model.RANSACRegressor()
    model.fit(X, Y)
    # pxs = np.arange(0, cols).reshape(1,-1).T
    pxs = np.array([0, cols]).reshape(1, -1).T
    pys = model.predict(pxs)
    axis = np.array([[0, pys[0]], [cols, pys[-1]]])
    return axis


@timer
def predict(img, Gs):
    assert len(Gs) > 0

    cs, rs = [], []
    rows, cols = img.shape[1], img.shape[2]
    for G in Gs:
        for _ in range(GAMMA):
            rx = np.random.randint(RADIUS, cols - RADIUS)
            ry = np.random.randint(RADIUS, rows - RADIUS)

            c, r = find_center(img, G, (rx, ry))
            if c is not None:
                cs.append(c), rs.append(r)
    cs = np.array(cs)
    axis = ransac(cs, rows, cols)
    return cs, rs, axis


def get_inner_outer_circle(cs, rs, axis):
    ERROR = 10
    rc = np.array([[r, c[0], c[1]] for r,c in zip(rs, cs) if distance(c, axis) < ERROR])
    min = np.argmin(rc[:,0])
    max = np.argmax(rc[:,0])
    c1, c2 = rc[min, 1:], rc[max, 1:]
    c1, c2 = closest_point(c1, axis), closest_point(c2, axis)
    return c1, rc[min, 0], c2, rc[max, 0]


def estimate(img, Gs):
    cs, rs, axis = predict(img, Gs)
    inner, r_inner, outer, r_outer = get_inner_outer_circle(cs, rs, axis)

    if VERBOSE:
        cs = np.array(cs)
        plt.imshow(sanitize(img))
        plt.plot(cs[:, 0], cs[:, 1], 'o', markersize=2)
        plt.plot(axis[:, 0], axis[:, 1])
        if inner is not None:
            plt.plot(inner[0], inner[1], '+', color='black')
            plt.text(inner[0], inner[1], 's')
        if outer is not None:
            plt.plot(outer[0], outer[1], '+', color='black')
            plt.text(outer[0], outer[1], 'e')
        plt.show()

    return inner, r_inner, outer, r_outer
