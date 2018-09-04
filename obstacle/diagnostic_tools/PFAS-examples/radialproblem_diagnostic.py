from test_diagnostic import pfas_test
import numpy as np

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

coarse_mx = 1
coarse_my = 2
num_cycles = 6
cycle = 'V'
x1, x2, y1, y2 = -2.0, 2.0, -2.0, 2.0
bounds = (x1, x2, y1, y2)
mx, my = coarse_mx, coarse_my
for j in range(num_cycles - 1):
    mx = 2 * mx + 1
    my = 2 * my + 1
print(mx)
N = (mx + 2)*(my + 2)
u0 = np.zeros(N)
kk = lambda i, j: (mx + 2)*j + i
X = np.linspace(x1, x2, mx + 2)
Y = np.linspace(y1, y2, my + 2)
for i in range(my + 2):
    for j in range(mx + 2):
        r = np.sqrt(X[j]**2 + Y[i]**2)
        if r > .69797 and X[j] not in [-2.0,2.0] and Y[i] not in [-2.0,2.0]:
            u0[kk(i, j)] = g(X[j],Y[i])

pfas_test(bounds, f, g, psi, cycle, uexact = uexact, coarse_mx = coarse_mx,\
                                coarse_my = coarse_my, min_num_levels = num_cycles, u0=u0)
