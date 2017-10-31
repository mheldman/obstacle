from multigrid import fmg
from Poisson2D import Poisson2D
import numpy as np

f = lambda x,y: -2.0*((1 - 6.0*x**2)*(y**2)*(1.0 - y**2) + (1.0 - 6.0*y**2)*(x**2)*(1.0 - x**2))
g = lambda x,y: x*0
minm = 2
numcycles = 7
m = minm
for i in range(0, numcycles):
    m = 2 * m + 1
N = (m + 2)**2
A, U, F, _, X = Poisson2D(m, f, bvals = True)
U = fmg(m, F, numcycles = numcycles, eta1 = 1, eta2 = 3)

Uexact = np.zeros((N, 1))
uexact = lambda x,y: (x**2 - x**4)*(y**4 - y**2)
kk = lambda i,j: j * (m + 2) + i
for j in range(0, m + 2):
    for i in range(0, m + 2):
        k = kk(i,j)
        Uexact[k] = uexact(X[i],X[j])
print(np.linalg.norm(U - Uexact, np.inf))