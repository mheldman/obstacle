import numpy as np

x1, x2, y1, y2 = 0.0, 1.0, 0.0, 1.0
psi = lambda x, y:  -np.minimum(np.minimum(x, y), np.minimum(1.0-x, 1.0-y))
f = lambda x, y: 8.0*np.ones(len(y))
g = lambda x, y: 0.*y
bounds = (x1, x2, y1, y2)
uexact = None
