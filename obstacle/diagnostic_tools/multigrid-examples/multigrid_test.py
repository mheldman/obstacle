from test import multigrid_test
from numpy import pi, sin

f = lambda x, y: -8.0*(pi**2)*sin(2.0*pi*x)*sin(2.0*pi*y)
g = lambda x, y: 0


x1, x2, y1, y2 = -1.0, 1.0, -1.0, 1.0
bounds = (x1, x2, y1, y2)
cycle = 'FV'
uexact = lambda x,y: sin(2.0*pi*x)*sin(2.0*pi*y)
coarse_mx = 5
coarse_my = 5
num_cycles = 2

multigrid_test(bounds, f, g, cycle, uexact=uexact, coarse_mx=coarse_mx,\
                                coarse_my=coarse_my, min_num_levels=num_cycles)
