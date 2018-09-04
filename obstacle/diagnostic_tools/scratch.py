import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def psi(x, y):
    z = np.sqrt(np.maximum(1 - x ** 2 - y ** 2, 0.0))
    if type(x) in [float, int]:
        if z < 1/np.sqrt(2):
            return -(x**2 + y**2)/np.sqrt(2) + np.sqrt(2) - 1/(2*np.sqrt(2))
    else:
        z[[z < 1 / np.sqrt(2)]] = -(x[[z < 1 / np.sqrt(2)]] ** 2 + y[[z < 1 / np.sqrt(2)]] ** 2) / np.sqrt(2) \
                                            + np.sqrt(2) - 1 / (2 * np.sqrt(2))
    return z
def psi(x, y):
    x = np.array([x])
    y = np.array([y])
    z = np.sqrt(np.maximum(1 - x ** 2 - y ** 2, 0.0))
    z[[z < 1 / np.sqrt(2)]] = -(x[[z < 1 / np.sqrt(2)]] ** 2 + y[[z < 1 / np.sqrt(2)]] ** 2) / np.sqrt(2) \
                                            + np.sqrt(2) - 1 / (2 * np.sqrt(2))
    return z[0]
print(psi(1,2))
X = np.linspace(-2.0,2.0,500)
Y = np.linspace(-2.0,2.0,500)
A, B = np.meshgrid(X, Y)
C = psi(A, B)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf1 = ax.plot_surface(A,B,C,color = 'r',vmin = 0.0, vmax = 5.1, alpha = .5)
plt.show()

