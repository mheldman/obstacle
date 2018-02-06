from test import rsp_test
import numpy as np

Alpha = .68026
Beta = .47152
psi = lambda x, y: np.sqrt(np.maximum(0.0, 1 - x**2 - y**2)) + np.minimum(0.0, 1 - x**2 - y**2)
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
rsp_test(bounds, f, g, psi, 400, 400, 'splu', uexact = uexact)