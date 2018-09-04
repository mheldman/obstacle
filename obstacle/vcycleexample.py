from multigrid import vcycle
from Poisson2D import poisson2d, rhs
import numpy as np

f = lambda x,y: -8.0*(np.pi**2)*np.sin(2.0*np.pi*x)*np.sin(2.0*np.pi*y)
minm = 1
numcycles = 6
m = minm
for i in range(0, numcycles):
    m = 2 * m + 1
N = (m + 2)**2
A = poisson2d(m)
F = rhs(m, f)
U = np.zeros(N)
U = vcycle(m, U, A, F, numcycles = numcycles, eta1 = 3, eta2 = 3)
X = np.linspace(-1.0, 1.0, m + 2)
Uexact = np.zeros(N)
uexact = lambda x,y: np.sin(2.0*np.pi*x)*np.sin(2.0*np.pi*y)
kk = lambda i,j: j * (m + 2) + i
for j in range(0, m + 2):
    for i in range(0, m + 2):
        k = kk(i,j)
        Uexact[k] = uexact(X[i],X[j])
print(np.linalg.norm(Uexact - U, np.inf))