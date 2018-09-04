from test_diagnostic import rsp_test, pfas_test
import numpy as np
import matplotlib.pyplot as plt


Alpha = .68026
Beta = .47152
def psi(x, y):
    x = np.array([x])
    y = np.array([y])
    z = np.sqrt(np.maximum(1 - x ** 2 - y ** 2, 0.0))
    z[[z < 1 / np.sqrt(2)]] = -(x[[z < 1 / np.sqrt(2)]] ** 2 + y[[z < 1 / np.sqrt(2)]] ** 2) / np.sqrt(2) \
                                            + np.sqrt(2) - 1 / (2 * np.sqrt(2))
    return z[0]
f = lambda x, y: 0.0
g = lambda x, y: -Alpha*np.log(np.sqrt(x**2 + y**2)) + Beta

def uexact(x, y):
    r = np.sqrt(x**2 + y**2)
    if r > .69797:
        return g(x, y)
    else:
        return psi(x, y)

x1, x2, y1, y2 = -2.0, 2.0, -2.0, 2.0
bounds = (x1, x2, y1, y2)
mx = 500
my = 500
N = (mx + 2)*(my + 2)
u0 = np.zeros(N)
kk = lambda i, j: (mx + 2)*j + i
X = np.linspace(x1, x2, mx + 2)
Y = np.linspace(y1, y2, my + 2)
for i in range(my + 2):
    for j in range(mx + 2):
        r = np.sqrt(X[j]**2 + Y[i]**2)
        if r < .69797:
            u0[kk(j, i)] = psi(X[j], Y[i])
        elif X[j] in [-2.0, 2.0] or Y[i] in [-2.0, 2.0]:
            u0[kk(j, i)] = g(X[j], Y[i])
        else:
            u0[kk(j, i)] = 0.0

U = rsp_test(bounds, f, g, psi, mx, my, 'splu', uexact = uexact)

#U = pfas_test(bounds, f, g, psi, 'V', uexact = uexact, coarse_mx = 1,\
        #                        coarse_my = 1, min_num_levels = 6, u0=U)

plt.close()
X = np.linspace(-2.0,2.0,mx+2)
Y = np.linspace(-2.0,2.0,my+2)
A, B = np.meshgrid(X, Y)
C = psi(A, B)

Z = np.zeros_like(C)
for i in range(my + 2):
    for j in range(mx + 2):
        Z[i, j] = U[kk(j, i)]
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.ioff()
fig = plt.figure()
ax = fig.gca(projection='3d')
surf1 = ax.plot_surface(A, B, C, color = 'b',vmin = -.5, vmax = 1.0, alpha = 1)
surf2 = ax.plot_surface(A, B, Z, color = 'r',vmin = -.5, vmax = 1.0, alpha = .35, cmap='viridis')
ax.set_zlim(-.5, 1)
plt.pause(5)
plt.show()