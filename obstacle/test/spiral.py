import numpy as np
from numpy import pi, sqrt, arctan2, sin

x1, x2, y1, y2 = -1., 1., -1., 1.

def psi(x, y):
  P = np.zeros_like(x)
  r = x*x + y*y
  v1 = (r > 1e-14)
  v2 = (r < 1e-14)
  r = sqrt(r[v1])
  P[v1] = sin(2.*pi/r + pi/2. - arctan2(x[v1], y[v1])) + r*(r + 1.)/(r - 2.) - 3.*r + 3.6
  P[v2] = 3.6
  return P


f = lambda x, y: 0.*y
g = lambda x, y: 0.*y
bounds = (x1, x2, y1, y2)
uexact = None
