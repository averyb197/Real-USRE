import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time
import astyle
astyle.make_pretty()

@njit(nogil=True)
def mandyp(c, n):
    z = c
    for i in range(n):
        if abs(z) > 2:
            return i
        z = z**2 + c

    return n

@njit(nogil=True)
def mandy_set(xmn, xmx, ymn, ymx, res=100, n=100):
    x = np.linspace(xmn, xmx, res)
    y = np.linspace(ymn, ymx, res)
    z = np.empty((res, res))

    for i in range(res):
        for j in range(res):
            z[i, j] = mandyp(x[i] + 1j * y[j], n) # 1j create complex with real part 0, imaginary part 1 so x[i] + 1j y[j] creates full complex number plane with loop

    return z


def plot_mandy(res=100):
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5

    start=time.time()
    Z = mandy_set(xmin, xmax, ymin, ymax, res)
    #5.53 vs .77 , res 1000,
    end = time.time()

    print(end-start)

    plt.imshow(Z.T, cmap="plasma")
    plt.show()

def mandy3d():
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    Z = mandy_set(xmin, xmax, ymin, ymax, res=1000)
    plt.scatter()


def logistic():
    interval = (2.8, 4)  # start, end
    accuracy = 0.0001
    reps = 600  # number of repetitions
    numtoplot = 200
    lims = np.zeros(reps)

    fig, biax = plt.subplots()
    lims[0] = np.random.rand()
    for r in np.arange(interval[0], interval[1], accuracy):
        for i in range(reps - 1):
            lims[i + 1] = r * lims[i] * (1 - lims[i])

        biax.plot([r] * numtoplot, lims[reps - numtoplot :], "b.", markersize=0.02)

    fig.set_size_inches(10, 6)
    biax.set(xlabel="r", ylabel="x", title="logistic map")
    plt.show()

logistic()


