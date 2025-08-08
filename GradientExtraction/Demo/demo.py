import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate
from Radial.helper import color_grads, read_img, radius


def cone():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    N = 10
    rings = np.array([
        [[(N-i)*np.cos(t), (N-i)*np.sin(t), i] for t in np.arange(0, 2*np.pi+2*np.pi/100,  2*np.pi/100)] for i in range(N)
    ])

    for ring in rings:
        ax.plot3D(ring[:,0], ring[:,1], ring[:,2], linewidth=5, cmap='plasma')

    plt.show()


def cylinder_():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    N = 10
    rings = np.array([
        [[np.cos(t), np.sin(t), i] for t in
         np.arange(0, 2 * np.pi + 2 * np.pi / 100, 2 * np.pi / 100)] for i in range(N)
    ])

    for ring in rings:
        ax.plot3D(ring[:, 0], ring[:, 1], ring[:, 2], linewidth=5)

    plt.show()


def eccentric_cone():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    N = 10

    rings = []
    for i in range(N):
        cix, ciy = 0.5*(N-i),  0.5*(N-i)
        ring = np.array([[(N-i)*np.cos(t) - cix, (N-i)*np.sin(t)-ciy, i] for t in
         np.arange(0, 2 * np.pi + 2 * np.pi / 100, 2 * np.pi / 100)])
        rings.append(ring)

    rings = np.array(rings)
    for ring in rings:
        ax.plot3D(ring[:, 0], ring[:, 1], ring[:, 2], linewidth=5)

    plt.show()


def plane_():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    N = 10

    rings = []
    for i in range(N):
        rings.append([[-N*0.5, 0, i], [N*0.5, 0, i]])
    rings = np.array(rings)
    for ring in rings:
        ax.plot3D(ring[:, 0], ring[:, 1], ring[:, 2], linewidth=5)

    plt.show()


def concentric():
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    # Make data.
    X = np.arange(-10, 10, 0.1)
    Y = np.arange(-10, 10, 0.1)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.clip(9 - R, a_min=0, a_max=10)  #np.sin(R)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma,
                           linewidth=0, antialiased=False)
    surf1 = ax2.imshow(Z, cmap=cm.plasma)
    fig.colorbar(surf, shrink=0.5, aspect=6)
    plt.show()


def eccentric():
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(1,2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    # Make data.
    X = np.arange(-10, 10, 0.1)
    Y = np.arange(-10, 10, 0.1)
    X, Y = np.meshgrid(X, Y)
    shape = X.shape

    XY = np.stack([X.flatten(), Y.flatten()]).T
    def radius():
        op = (o[None, :] - XY)
        a = (e.T @ e - 1)
        b = 2 * (e[None, :] * op).sum(axis=1)
        c = (op * op).sum(axis=1)
        dis = np.clip(-(4 * a * c) + (b ** 2), a_min=0, a_max=np.inf)
        return (-b - np.sqrt(dis)) / (2 * a)

    e = np.array([0.5, 0])
    o = np.array([0,0])
    R = radius()
    R = R.reshape(shape)

    # R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.clip(8 - R, a_min=0, a_max=10)  # np.sin(R)


    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=0, antialiased=False)
    surf1 = ax2.imshow(Z, cmap=cm.plasma)
    fig.colorbar(surf, shrink=0.5, aspect=6)
    plt.show()


def cylinder():
    def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
        z = np.linspace(0, height_z, 50)
        theta = np.linspace(0, 2 * np.pi, 50)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid) + center_x
        y_grid = radius * np.sin(theta_grid) + center_y
        return x_grid, y_grid, z_grid

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    Xc, Yc, Zc = data_for_cylinder_along_z(0.2, 0.2, 0.05, 0.1)
    surf = ax.plot_surface(Xc, Yc, Zc, cmap=cm.plasma)
    fig.colorbar(surf, shrink=0.5, aspect=6)

    plt.show()


def plane():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X = np.arange(-10, 10, 0.1)
    Y = np.arange(-10, 10, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = Y

    surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=6)
    plt.show()


def eccentric_circles():
    from misc import get_circle
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    f = np.zeros(2)
    e = np.ones(2)*0.5
    ax.plot(f[0], f[1], 'bo')
    axis = [f]
    for i in range(12):
        r_i = i+1
        f_i = f + r_i * e
        c_i = get_circle(f_i, r_i, 1, 0)
        if i < 10:
            ax.plot(c_i[:,0], c_i[:,1], 'b-')
        else:
            ax.plot(c_i[:, 0], c_i[:, 1], 'b--')
        axis.append(f_i)
    axis = np.array(axis)
    ax.plot(axis[:,0], axis[:,1], 'ro-')
    l = 20
    ax.plot([axis[-1,0], axis[-1,0] + l*e[0]], [axis[-1,1], axis[-1,1] + l*e[1]], 'r--')
    plt.show()


if __name__ == '__main__':
    # eccentric_circles()
    concentric()
    eccentric()
    concentric()
    # cylinder()
    # plane()

    # Step
    # Polynomial / Exponential blending