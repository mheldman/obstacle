import numpy as np
from numpy import pi, sqrt, arctan, sin

mini = lambda x, y: np.minimum(x, y)
x1, x2, y1, y2 = -1.0, 1.0, -1.0, 1.0
psi = lambda x, y: -(x**2 - 1.)*(y**2 - 1.)
f = lambda x, y: 2.*(x**2 + y**2 - 2)
g = lambda x, y: 0.*y
bounds = (x1, x2, y1, y2)
uexact = None
