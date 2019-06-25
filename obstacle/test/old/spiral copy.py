import numpy as np
from numpy import pi, sqrt, arctan2, sin

mini = lambda x, y: np.minimum(x, y)
x1, x2, y1, y2 = 0.0, 1.0, 0.0, 1.0
psi = lambda x, y: -.5*x
f = lambda x, y: -1.0
g = lambda x, y: 0.*y
bounds = (x1, x2, y1, y2)
uexact = None
