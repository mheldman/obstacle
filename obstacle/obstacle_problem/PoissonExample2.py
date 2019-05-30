'''
Example solution of the Poisson equation. The example uses the default settings with f(x,y) = -8*pi^2*sin(2*pi*x)sin(2*pi*y) and exact solution u(x,y) = sin(2*pi*x)sin(2*pi*y).
'''

import numpy as np
from Poisson2D import Poisson2D

f = lambda x,y: -8.0*(np.pi**2)*np.sin(2.0*np.pi*x)*np.sin(2.0*np.pi*y)
m = 50
A, U, P, F, X = Poisson2D(m, f)
Uexact = np.zeros((m**2,1))
h = 2 / (m + 1)
U = np.linalg.solve(A, F)
uexact = lambda x,y: np.sin(2.0*np.pi*x)*np.sin(2.0*np.pi*y)

kk = lambda i, j: j * m + i
for j in range(0, m):
    for i in range(0, m):
        k = kk(i,j)
        Uexact[k] = uexact(X[i], X[j])
print(np.linalg.norm(Uexact - U, np.inf))