import numpy as np
from obstacle.obstacle_problem import box_obstacle_problem
from obstacle.multigrid.GS import pgs
from scipy.sparse.linalg import spsolve

rstar = .6979651482

def psi(x, y):
    x = np.array([x])
    y = np.array([y])
    z = np.sqrt(np.maximum(1 - x ** 2 - y ** 2, 0.0))
    z[[z < 1 / np.sqrt(2)]] = -(x[[z < 1 / np.sqrt(2)]] ** 2 + y[[z < 1 / np.sqrt(2)]] ** 2) / np.sqrt(2) \
                              + np.sqrt(2) - 1 / (2 * np.sqrt(2))
    return -100.
    #return z[0]

f = lambda x, y: 0.0
g = lambda x, y: -rstar ** 2 * np.log(np.sqrt(x ** 2 + y ** 2) / 2.) / np.sqrt(1. - rstar**2)
x1, x2, y1, y2 = -2.0, 2.0, -2.0, 2.0
bounds = (x1, x2, y1, y2)

def uexact(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    cond1 = (r > rstar)
    cond2 = ~cond1
    Uexact = 0.*x
    Uexact[cond1] = g(x[cond1], y[cond1])
    Uexact[cond2] = psi(x[cond2], y[cond2])
    return Uexact

mx = 400
my = mx


radial = box_obstacle_problem(bounds, f, g, psi)
radial.discretize(mx, my)
#radial.initialize()
A, F = radial.A, radial.F

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = (mx + 2)*(my + 2)
Uexact = np.zeros(N)
X = np.linspace(x1, x2, mx + 2)
Y = np.linspace(y1, y2, my + 2)
[X, Y] = np.meshgrid(X, Y)
Uexact = uexact(X.flatten(), Y.flatten())
#Uexact = spsolve(A, F)
noise = 5.*np.sin((1/2)*np.pi*X.flatten())*np.sin((1/2)*np.pi*Y.flatten()) + 5.*np.sin(16.*np.pi*X.flatten())*np.sin(16.*np.pi*Y.flatten()) + 5.*np.sin(16.*np.pi*X.flatten())*np.sin(90.*np.pi*Y.flatten())
#Uexact = Uexact.reshape((mx + 2, my + 2))
U = Uexact + noise# - radial.P
U[U < 0.] = 0.

for i in range(20):
  
  pgs(A, U, F, maxiters=2)
  print(np.linalg.norm(U - (Uexact - radial.P), np.inf))
  Error = U - Uexact + radial.P
  Z = Error.reshape((mx + 2, my + 2))
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(X, Y, Z)
  plt.show()
