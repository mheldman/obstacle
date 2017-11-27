from test import pfas_test
import numpy as np
f = lambda x, y: -8.0*(np.pi**2)*np.sin(2.0*np.pi*x)*np.sin(2.0*np.pi*y)
g = lambda x, y: 0
psi = lambda x, y: -10**8


x1, x2, y1, y2 = -1.0, 1.0, -1.0, 1.0
bounds = (x1, x2, y1, y2)
cycle = 'FV'
uexact = lambda x,y: np.sin(2.0*np.pi*x)*np.sin(2.0*np.pi*y)
coarse_mx = 1
coarse_my = 1
num_cycles = 6

multigrid_test(bounds, f, g, psi, cycle, uexact=uexact, coarse_mx=coarse_mx,\
                                coarse_my=coarse_my, min_num_levels=num_cycles)
