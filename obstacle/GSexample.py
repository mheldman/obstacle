import numpy as np
from Poisson2D import Poisson2D
from GS import gs

'''
Example solution of AU = F using the Gauss-Seidel iteration. Here A is the discrete Poisson matrix and F is the discrete version of the function f(x,y) = -8.0*(np.pi**2)*np.sin(2.0*np.pi*x)*np.sin(2.0*np.pi*y).
'''

f = lambda x,y: -8.0*(np.pi**2)*np.sin(2.0*np.pi*x)*np.sin(2.0*np.pi*y)
m = 100
A, U, F, P, X = Poisson2D(m,f)
Uexact = np.zeros(((m+2)**2,1))
h = 2/(m + 1)
U = gs(U, A, F, (m + 2)**2, maxiters = 300, tol = 10**-8)
X = np.linspace(-1.0, 1.0, m+2)
uexact = lambda x,y: np.sin(2.0*np.pi*x)*np.sin(2.0*np.pi*y)
kk = lambda i,j: j * (m + 2) + i

for j in range(0, m + 2):
    for i in range(0, m + 2):
        k = kk(i,j)
        Uexact[k] = uexact(X[i],X[j])

print(np.linalg.norm(Uexact - U, np.inf))