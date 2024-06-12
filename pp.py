import numpy as np
import matplotlib.pyplot as plt
from ODE_TOOLS import make_VF
from numba import njit
import astyle
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
astyle.make_pretty()
@njit
def pp01(interval, targs, dh=1e-3):
    n = int(interval/dh)
    t = np.linspace(0, interval, n)

    X = np.empty(n)
    Y = np.empty(n)

    # X0 = 10
    # Y0 = 13

    K, r, b, h, a, d, sigma, X0, Y0 = targs

    X[0] = X0
    Y[0] = Y0

    # K = 20 # carrying capacity of population
    # r = .2  # growing rate of prey pop
    # b = .001#searching rate
    # h = .5 # h = handling time
    # a = 2 #Allee param - benefit of increased population, kinda like how much given species needs to cooperate, life expectancy is often low when population is low even with
    # #plenty of resources
    # d = .05 # mortality for y
    # sigma = .02 # rate at which surplus prey is converted into predator pop



    dxdt = lambda x, y: (r * x * (x - a) * (1 - (x/K))) - (b * x * y)
    dydt = lambda x, y: ((sigma * b * x * y)/(1 + (b * h * x))) - (d * y)

    for m in range(n-1):
        X[m+1] = X[m] + dh * dxdt(X[m], Y[m])
        Y[m+1] = Y[m] + dh * dydt(X[m], Y[m])

    return t, X, Y
def stability_analysis(x, y, K=44, r=.12, b=.12, h=.99, d=.315, a=.66, sigma=.53):
    eq1 = (0, 0)
    eq2 = (a, 0)
    eq3 = (K, 0)
    q = d / (b * (sigma - d * h))
    eq4y = (r / (b * K)) * (q - a) * (K - q)
    eq4 = (q, eq4y)

    eq_points = [eq1, eq2, eq3, eq4]


    def Jacobian(x, y):
        return np.array([
        [(2 * r * x) - ( (3 * r * x**2)/ K ) - (r * a) - (b * y) + ( (2 * r * a * x)/K ),
         (-b * K)
        ],
        [(sigma * b * y)/(b*x*h + 1)**2 ,
         (sigma * b * x)/(1 + b * h * x) - d
        ]
        ])

    print(f"K = {K}\n"
          f"r = {r}\n"
          f"b = {b}\n"
          f"h = {h}\n"
          f"d = {d}\n"
          f"a = {a}\n"
          f"sigma = {sigma}\n")

    for point in eq_points:
        l1, l2 = np.linalg.eig(Jacobian(point[0], point[1]))[0]



        print(type(l1), type(l2), "\n")

        typo = -1

        print(f" Point ({point[0]:.3f}, {point[1]:.3f}):\n"
              f" -- Eigen Values: ({l1:.3f}, {l2:.3f})\n"
              f" -- Type: {typo}")

def plot_nc(targs):
    K, r, b, h, a, d, sigma = targs[:-2]
    q = d / (b * (sigma - d * h))

    x = np.linspace(0, K)
    y = (r/b) * (x-a) * (1 - x/K)
    plt.plot(x, y, c="white")
    plt.axvline(q, 0,100, c="white")

def plot_pp01(rand=False):
    X0 = 95.0#q
    Y0 = 20.0 #eq4y

    if rand:
        while True:
            K = float(np.random.randint(1, 50))
            r = np.random.rand()
            b = np.random.rand()
            h = np.random.rand()
            a = np.random.rand()
            d = np.random.rand()
            sigma = np.random.rand()

            q = d/(b * (sigma - d*h))

            if q>0 and a<q and q<K:
              #  targs = [K, r, b, h, a, h, sigma, X0, Y0]
                print(f"K = {K}\n"
                      f"r = {r}\n"
                      f"b = {b}\n"
                      f"h = {h}\n"
                      f"a = {a}\n"
                      f"d = {d}\n"
                      f"sigma = {sigma}")
                break

    else:
        K = 38.0
        r = 0.8360346534665105
        b = 0.06042445758166415
        h = 0.5034374369716825
        a = 0.17891038414771132
        d = 0.23463094578034538
        sigma = 0.8708806437827094
        q = d / (b * (sigma - d * h))

    targs = [K, r, b, h, a, d, sigma, X0, Y0]

    eq1 = (0, 0)
    eq2 = (a, 0)
    eq3 = (K, 0)

    eq4y = (r/(b*K)) * (q - a) * (K - q)
    eq4 = (q, eq4y)
    eqx, eqy = zip(eq1, eq2, eq3, eq4)

   # targs = [K, r, b, h, a, h, sigma, X0, Y0]

    print(f"q={q}")
    print(f"a<q<K: {a<q and q<K}\n"
          f"q > 0: {q>0}")

    stability_analysis(X0, Y0, K=K, r=r, b=b, h=h, sigma=sigma, a=a, d=d)

    fig, ax = plt.subplots(2, figsize=(10, 8))
    interval=70
    t, X, Y = pp01(interval, targs)
    print(X)

    num=10

    xran = np.linspace(0, K+50, num+20)
    yran = np.linspace(0, K+50, num)

    for k in range(len(xran)):
        targs[-2] = xran[k]
        for j in range(len(yran)):
            targs[-1] = yran[j]
            t, X, Y = pp01(interval, targs)
            plt.scatter(xran[k], yran[j], c="red", s=1)
            plt.plot(X, Y)

    num_intr = 10
    poix = np.linspace(q-0.02, q+0.02, num_intr)
    poiy = np.linspace(eq4y-.1, eq4y+.1, num_intr)

    for k in range(len(poix)):
        targs[-2] = poix[k]
        for j in range(len(poiy)):
            targs[-1] = poiy[j]
            t, X, Y = pp01(interval, targs)
            plt.scatter(poix[k], poiy[j], c="red", s=1)
            plt.plot(X, Y)


    plot_nc(targs)


    ax[0].plot(t, X, c="#FF00FF")
    ax[0].plot(t, Y, c="#41FEFF")
    ax[0].legend(["X", "Y"])
    ax[1].scatter(eqx, eqy, s=20, c="red")

    plt.show()

plot_pp01()

def make_params(n=10, kmax=30, rmax=5, bmax=1, hmax=1, dmax=1, amax=30, sigmamax=1, epsilon=1e-3):
    K = np.linspace(1, kmax, n)
    r = np.linspace(epsilon, rmax, n)
    b = np.linspace(epsilon, bmax, n)
    h = np.linspace(epsilon, hmax, n)
    d = np.linspace(epsilon, dmax, n)
    a = np.linspace(epsilon, amax, n)
    sigma = np.linspace(epsilon, sigmamax, n)


    q = d/(b * (sigma - d*h))

    valid_params = []

    for i in range(n):
        if q[i] > 0 and (a[i]<q[i]<K[i]):
            valid_params.append(np.array([K, r, b, h, d, a, sigma, q]))

    print(valid_params)

def ani_henson():
    fig, ax = plt.subplots(figsize=(8, 6))

    line, = ax.plot([], [], 'b-', lw=2)
    point, = ax.plot([], [], 'ro')
    time_template = 'Time = %.1f'
    curr_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    r_vals=np.linspace(.001, 4, 100)

    K = 20.0
    r = 1.0
    b = .01
    h = .5
    a = 4.0
    d = .05
    sigma = .02

    eq1 = (0, 0)
    eq2 = (a, 0)
    eq3 = (K, 0)
    q = d/(b * (sigma - d*h))
    eq4y = (r/(b*K)) * (q - a) * (K - q)
    eq4 = (q, eq4y)
    eqx, eqy = zip(eq1, eq2, eq3, eq4)

    X0 = 10.0#q
    Y0 = 13.0 #eq4y

    #targs = [K, r, b, h, a, h, sigma, X0, Y0]

    num = 20

   # @njit

    def make_trajectories():
        interval = 30
        trajectories = []
        for curr_r in r_vals:
            targs = [K, curr_r, b, h, a, d, sigma, X0, Y0]
            xran = np.linspace(0, K + 50, num)
            yran = np.linspace(0, K + 50, num)
            trajectories_for_r = []
            for x0 in xran:
                targs[-2] = x0
                for y0 in yran:
                    targs[-1] = y0
                    t, X, Y = pp01(interval, targs)
                    trajectories_for_r.append((X, Y))
            trajectories.append(trajectories_for_r)
        return trajectories

    lines = [ax.plot([], [])[0] for _ in range(num)]
    time_template = 'r = %.2f'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        point.set_data([], [])
        curr_text.set_text('')
        return line, point, curr_text

    trajectories = make_trajectories()
    #print(trajectories)

    def animate(i):
        r = r_vals[i]
        for line, traj in zip(lines, trajectories[i]):
            line.set_data(traj[0], traj[1])
        time_text.set_text(time_template % r)
        return lines + [time_text]

    ani = animation.FuncAnimation(fig, animate, frames=len(r_vals), init_func=init,
                                  interval=200, blit=True)

    return ani


#stability_analysis(0, 0)





















def lotka_volterra(interval=12, alpha=1.1, beta=.4, delta=.1, gamma=.4, X0=1, Y0=1, h=1e-3): # HELLA SENSITIVE TO h
    n = int(interval/h)

    t = np.linspace(0, interval, n)

    X = np.empty(n)
    Y = np.empty(n)

    X[0] = X0
    Y[0] = Y0

    dxdt = lambda x, y: (alpha * x) - (beta * x * y)
    dydt = lambda x, y: (delta * x * y) - (gamma * y)

    for k in range(n-1):
        X[k+1] = X[k] + h * dxdt(X[k], Y[k])
        Y[k+1] = Y[k] + h * dydt(X[k], Y[k])

    return t, X, Y

def lotka_volterra_phase_plane():
    steps = np.linspace(.9, 1.8, 10)
    print(steps)
    for i in range(len(steps)):
        t, X, Y = lotka_volterra(X0=steps[i], Y0=steps[i])
        plt.plot(X, Y)


    t, X, Y = lotka_volterra(X0=.9, Y0=.9)
    plt.plot(X, Y)
    plt.legend([f"X0=Y0={i:.1f}" for i in steps])

    # plt.plot(t, X)
    # plt.plot(t, Y)
    # plt.legend(["X", "Y"])
    plt.show()





