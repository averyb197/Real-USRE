import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit

sns.set()


#@njit
def make_VF(fx, fy, interval, num_vec=50, norm_fac=.75):  # use plt.quiver directly with X, Y, U, V
    t = np.linspace(0, interval, num_vec)
    X, Y = np.meshgrid(t, t)
    U = fx(X, Y)
    V = fy(X, Y)
    mags = np.sqrt(U ** 2 + V ** 2)
    U = (U/mags) * norm_fac
    V = (V/mags) * norm_fac

    return X, Y, U, V


@njit
def forward_euler(f, interval=5, h=1e-3, t0=0,
                  y0=0):  # f is function of 2 variables (ie da DE), t is placeholder for any independent, y for dependent
    steps = int(interval / h)

    Y = np.zeros(steps)
    T = np.zeros(steps)

    Y[0] = y0
    T[0] = t0

    for i in range(steps - 1):
        # print(i)
        T[i + 1] = T[i] + h
        # print(f(T[i], Y[i]))
        Y[i + 1] = Y[i] + h * f(T[i], Y[i])

    return T, Y


def FDMdir(f, t0, tn, y0, yn, a=1, b=0, c=0, h=1e-1): # dirichilet boundary condtions
    n = int((tn - t0) / (h)) + 1  # discretize time/input domain
    t = np.linspace(t0, tn, n + 1)

    A = np.zeros((n - 1, n - 1))

    supd = 2*a + b*h  # superdiagnonal
    md = ((2 * h ** 2 * c) - (4 * a))  # main diagonal
    subd = 2*a - b*h  # subdiagnonal

    for i in range(0, n - 1):
        A[i, i] = md

    for i in range(0, n - 2):
        A[i, i + 1] = supd
        A[i + 1, i] = subd
    #     print(f(t[1:n-1]) * h**2)
    #     print(np.ones(n-1))
    v = f(t[1:n]) * 2 * h ** 2  # value vector, with value for each value of t in discretized time array

    v[0] = - subd * y0 + f(t[0]) * h ** 2
    v[-1] = - supd * yn + f(t[-1]) * h ** 2

    print(np.shape(v))

    y = np.linalg.solve(A, v)

    print(y0, y, yn)
    y = np.concatenate((y0, y, yn), axis=None)
    return t, y


def const(n):
    def rc(t):
        if type(t) != np.ndarray:
            return n
        return np.full(t.shape, n)
    return rc


#@njit
def FDMvn(f, t0, tn, yp0, ypn, a=const(1), b=const(0), c=const(0), h=1e-1):
    n = int((tn - t0) / h) + 1
    t = np.linspace(t0, tn, n + 1)

    A = np.zeros((n + 1, n + 1))

    supd = (2* a(t)) + (b(t) * h)  # superdiagnonal
    md = (2 * h**2 * c(t)) - (4 *a(t)) # main diagonal
    subd = (2*a(t)) - (b(t)*h)  # subdiagnonal

    for i in range(0, n + 1):
        A[i, i] = md[i]

    for i in range(0, n):
        A[i + 1, i] = subd[i]
        A[i, i + 1] = supd[i]

    A[0, 1] = 4 * a(t0)
    A[n, n - 1] = 4 * a(n)

    v = 2 * h ** 2 * f(t)
    v[0] += (2 * a(t0) - b(t0) * h) * (2 * h * yp0)
    v[-1] -= (2 * a(t0) - b(t0) * h) * (2 * h * ypn)

    print(n)
    print()
    print(A, A.shape, A[0, n])
    print()
    print(v)
    print()
    y = np.linalg.solve(A, v)
    return t, y

def tau(t):
    return t





