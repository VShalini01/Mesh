import numpy as np
import matplotlib.pyplot as plt

def Scale(s):
    return np.array([[1,0],[0,s]])

def Rotate(theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return R


def translate(ps, txy):
    return ps + txy


def draw():
    from misc import get_circle
    center = np.array([0, 0])
    circ_o = get_circle(center, 100, 1, 0)
    focal = np.array([0, 0])
    circ_f = get_circle(focal, 10, 1, 0)

    txy = np.array([100, 100])
    circ_o, center = translate(circ_o, txy), translate(center, txy)
    circ_f, focal = translate(circ_f, txy), translate(focal, txy)




    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.plot(circ_o[:,0], circ_o[:,1], 'r--')
    ax.plot(circ_f[:, 0], circ_f[:, 1], 'b--')
    ax.plot(focal[0], focal[1], 'o')
    ax.plot(center[0], center[1], '*')
    plt.show()


if __name__ == '__main__':
    draw()
