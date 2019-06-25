import numpy as np
from obstacle.obstacle_problem import box_obstacle_problem
from obstacle.multigrid.GS import pgs
from scipy.sparse.linalg import spsolve

import numpy as np
from numpy import pi, sqrt, arctan2, sin

mini = lambda x, y: np.minimum(x, y)
x1, x2, y1, y2 = -1.0, 1.0, -1.0, 1.0
def psi(x, y):
  P = np.zeros(len(x))
  r = x**2 + y**2
  v1 = (r != 0.)
  v2 = (r == 0.)
  r = sqrt(r[v1])
  P[v1] = sin(2*pi/r + pi/2 - arctan2(x[v1], y[v1])) + r*(r + 1)/(r - 2) - 3*r + 3.6
  P[v2] = 3.6
  return P

f = lambda x, y: 0.*y
g = lambda x, y: 0.*y
bounds = (x1, x2, y1, y2)
uexact = None

mx = 100
my = mx


radial = box_obstacle_problem(bounds, f, g, psi)
radial.discretize(mx, my)
#radial.initialize()
A, F = radial.A, radial.F

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


U = np.zeros((mx+2)*(my+2))

def energy(U):
  return np.inner(U, A.dot(U)) - np.inner(F, U)

for i in range(10):
  
  pgs(A, U, F, maxiters=100)
  print(energy(U))
