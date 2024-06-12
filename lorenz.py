import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import astyle
astyle.make_pretty()


sigma = 10
rho = 28
beta = 8 / 3

X0 = 0
Y0 = 1
Z0 = 1.5

interval = 10


@njit
def lorenz_fe(interval=10, h=1e-2):
    steps = int(interval / h)

    t = np.linspace(0, interval, steps)

    X = np.empty(steps)
    Y = np.empty(steps)
    Z = np.empty(steps)

    X[0] = X0
    Y[0] = Y0
    Z[0] = Z0

    dxdt = lambda x, y, z: sigma * (y - x)
    dydt = lambda x, y, z: (x * (rho - z)) - y
    dzdt = lambda x, y, z: (x * y) - (beta * z)

    for k in range(steps - 1):
        X[k + 1] = X[k] + h * dxdt(X[k], Y[k], Z[k])
        Y[k + 1] = Y[k] + h * dydt(X[k], Y[k], Z[k])
        Z[k + 1] = Z[k] + h * dzdt(X[k], Y[k], Z[k])

    return t, X, Y, Z


t, X, Y, Z = lorenz_fe(interval=interval)

#fig, lax = plt.subplots(1, 2, figsize=(10, 5))


def colorline(X, Y, t, cmap="plasma"):
    points = np.array([X, Y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    normed_t = Normalize(vmin=t.min(), vmax=t.max())
    line_collection = LineCollection(segments, array=t, norm=normed_t, cmap=cmap)

    return line_collection


def full_an():
    t, X, Y, Z = lorenz_fe(interval=interval)
    fog = plt.figure()
    aniax = fog.add_subplot(projection="3d")
    line, = aniax.plot([], [], [])

    aniax.set_xlim(np.min(X), np.max(X))
    aniax.set_ylim(np.min(Y), np.max(Y))
    aniax.set_zlim(np.min(Z), np.max(Z))

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return line,

    def update(frame):
        line.set_data(X[:frame], Y[:frame])
        line.set_3d_properties(Z[:frame])
        return line,

    ani = FuncAnimation(fog, update, frames=len(t), init_func=init, blit=True, interval=10)
    return ani


foa = full_an()
plt.show()
