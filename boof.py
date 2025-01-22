import numpy as np
import matplotlib.pyplot as plt
from pp import make_params, plot_nc
from numba import njit


def make_VF_good(n):
    x, y = np.meshgrid(np.linspace(0, 100, n), np.linspace(0, 100, n))
    return [x, y]

x, y = make_VF_good(400)
K, r, b, h, a, d, sigma = make_params(1)
u = (r * x * (x - a) * (1 - (x/K))) - (b * x * y)
v = ((sigma * b * x * y)/(1 + (b * h * x))) - (d * y)

# Plot the streamline
plt.figure(figsize=(8, 6))
plt.streamplot(x, y, u, v, color=np.sqrt(u**2 + v**2), cmap='viridis', density=4)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Phase Plane')
plt.grid()
ncx, ncy = plot_nc([K, r, b, h, a, d, sigma, 0, 0])
plt.ylim(0, max(ncy)+20)
plt.show()
