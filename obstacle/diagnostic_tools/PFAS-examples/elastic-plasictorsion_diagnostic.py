from numpy import linspace, minimum, array
import numpy as np
from test_diagnostic import pfas_test

coarse_mx = 1
coarse_my = 1
mx = coarse_mx
my = coarse_my
num_cycles = 7
for i in range(num_cycles):
    mx = 2*mx + 1
    my = 2*my + 1
N = (mx + 2)*(my + 2)

x1, x2, y1, y2 = 0.0, 1.0, 0.0, 1.0
w = linspace(x1, x2, 1000)
#w_list = [w for i in range(N)]
#w = np.array(w_list)

psi = lambda x, y: -min(min(x**2 + (y - w)**2), min((x-1)**2 + (y-w)**2), min((x - w)**2 + y**2), min((x-w)**2  + (y-1)**2))
f = lambda x, y: -5.0
g = lambda x, y: 0.0


cycle = 'V'
bounds = (x1, x2, y1, y2)
pfas_test(bounds, f, g, psi, cycle, coarse_mx = coarse_mx,\
                                coarse_my = coarse_my, min_num_levels = num_cycles)