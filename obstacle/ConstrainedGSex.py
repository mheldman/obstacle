'''
Example constrained Gauss-Seidel iteration on an obstacle problem.
'''
import numpy as np
from GS import gs
from Poisson2D import Poisson2D

Alpha = .68026
Beta = .47152
f = lambda x, y: x*0
g = lambda x, y: -Alpha*np.log(np.sqrt(x**2 + y**2)) + Beta  #exact solution
psi = lambda x, y: np.sqrt(np.maximum(0.0,1 - x**2 - y**2)) + np.minimum(0.0,1-x**2-y**2)
m = 50
N = m**2

A, U, P, F, X = Poisson2D(m, f, a = -2.0, b = 2.0, psi = psi, g = g)
U = gs(U, A, F, N, maxiters = 1000, tol = 10**-8, P = P)

h = 1/(m+1)
Uexact = np.zeros((N,1))
kk = lambda i, j: j * m + i
for j in range(m):
    for i in range(m):
        r = np.sqrt(X[i]**2 + X[j]**2)
        k = kk(i, j)
        if r > .69797:  #if (X[i], X[j]) is outside the contact region
            Uexact[k] = g(X[i], X[j])
        else:
            Uexact[k] = P[k]
        print(U[k], Uexact[k], X[i], X[j])
print('||U - Uexact||_inf =', max(list(abs(Uexact - U))))