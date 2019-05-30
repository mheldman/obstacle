'''
Example solution of the Poisson equation. The example uses the default settings with f(x,y) = -8*pi^2*sin(2*pi*x)sin(2*pi*y) and exact solution u(x,y) = sin(2*pi*x)sin(2*pi*y).
'''

import numpy as np
from numpy import meshgrid, linspace
from Poisson2D import poisson2d, rhs
from scipy.sparse.linalg import spsolve

f = lambda x,y: -8.0*(np.pi**2)*np.sin(2.0*np.pi*x)*np.sin(2.0*np.pi*y)
g = lambda x,y: 4*x*y
m = 100
A = poisson2d(m, bvals=False)
F = rhs(f, m, g=g, bvals=False)
Uexact = np.zeros((m+2)**2)
U = spsolve(A, F)
uexact = lambda x,y: np.sin(2.0*np.pi*x)*np.sin(2.0*np.pi*y) + 4*x*y
X = linspace(-1, 1, m+2)
kk = lambda i,j: j * (m + 2) + i
for j in range(0, m + 2):
    for i in range(0, m + 2):
        k = kk(i,j)
        Uexact[k] = uexact(X[i],X[j])

print(np.linalg.norm(Uexact - U, np.inf))
