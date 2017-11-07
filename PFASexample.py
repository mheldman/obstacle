
from multigrid import pfas
import numpy as np
from Poisson2D import Poisson2D
from GS import gs

Alpha = .68026
Beta = .47152
psi = lambda x,y: np.sqrt(np.maximum(0.0, 1 - x**2 - y**2)) + np.minimum(0.0, 1 - x**2 - y**2)
f = lambda x,y: x*0
g = lambda x,y: -Alpha*np.log(np.sqrt(x**2 + y**2)) + Beta

minm = 4
numcycles = 4
m = minm
for i in range(0, numcycles):
    m = 2 * m + 1
A, U, F, P, X = Poisson2D(m, f = f, psi=psi, a=-2.0, b=2.0, g = g, bvals = True)
F = F - A.dot(P)
U = U - P #use max(0.0, P) as an initial iterate for PFAS
h = 4/(m+1)
N = (m + 2)**2
Uexact = np.zeros(N)
kk = lambda i, j: j * (m + 2) + i
for j in range(m + 2):
    for i in range(m + 2):
        r = np.sqrt(X[i]**2 + X[j]**2)
        k = kk(i, j)
        if r > .69797:  #if (X[i], X[j]) is outside the contact region
            Uexact[k] = g(X[i], X[j])
            #print(U[k], Uexact[k], '(' + str(X[i]) + ',', str(X[j]) + ')')
        else:
            Uexact[k] = psi(X[i], X[j])
#U = Uexact - P #use exact solution as an initial iterate for PFAS
U = pfas(m, U, A, F, numcycles = numcycles, cyclenum = 0, eta = 1)
#U = gs(U, Ah, fh, (m+2)**2, maxiters = 1, P = True)
U = U + P
'''
Uexact = np.zeros(N)
kk = lambda i, j: j * (m + 2) + i
for j in range(m + 2):
    for i in range(m + 2):
        r = np.sqrt(X[i]**2 + X[j]**2)
        k = kk(i, j)
        if r > .69797:  #if (X[i], X[j]) is outside the contact region
            Uexact[k] = g(X[i], X[j])
            print(U[k], Uexact[k], '(' + str(X[i]) + ',', str(X[j]) + ')')
        else:
            Uexact[k] = psi(X[i], X[j])
'''

err = np.linalg.norm(Uexact - U, np.inf)
print('Error:', err)