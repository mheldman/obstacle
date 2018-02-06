from test import rsp_test
import numpy as np

f = lambda x, y: -8.0*(np.pi**2)*np.sin(2.0*np.pi*x)*np.sin(2.0*np.pi*y)
g = lambda x, y: 0
psi = lambda x, y: -10**8


x1, x2, y1, y2 = -1.0, 1.0, -1.0, 1.0
bounds = (x1, x2, y1, y2)
uexact = lambda x,y: np.sin(2.0*np.pi*x)*np.sin(2.0*np.pi*y)
print(rsp_test(bounds, f, g, psi, 100, 100, 'splu', uexact = uexact))
