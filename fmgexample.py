from multigrid import fmg
from Poisson2D import poisson2d
import numpy as np
from time import time
from Poisson2D import rhs

f = lambda x,y: -8.0*(np.pi**2)*np.sin(2.0*np.pi*x)*np.sin(2.0*np.pi*y)
minm = 1
for i in range(4,8):
    numcycles = i
    m = minm
    for i in range(0, numcycles):
        m = 2 * m + 1
    print('Fine grid size (' + str(m+2),'x',str(m+2) + ')')
    print(m**2, 'unknowns')
    N = (m + 2)**2
    A = poisson2d(m)
    F = rhs(m, f)
    tstart = time()
    U = fmg(m, F, numcycles = numcycles, eta1 = 3, eta2 = 3)
    timex = time() - tstart
    Uexact = np.zeros(N)
    X = np.linspace(-1.0,1.0,m+2)
    uexact = lambda x,y: np.sin(2.0*np.pi*x)*np.sin(2.0*np.pi*y)
    kk = lambda i,j: j * (m + 2) + i
    for j in range(0, m + 2):
        for i in range(0, m + 2):
            k = kk(i,j)
            Uexact[k] = uexact(X[i], X[j])
    print('Error: ||U - Uexact||_inf =', np.linalg.norm(U - Uexact, np.inf))
    print('Time for fmg:', timex, '\n')
