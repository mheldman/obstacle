import numpy as np
from numpy import zeros
from GS import gs
from Poisson2D import Poisson2D

def interpolate(U, m, F = zeros((1,1))):
    N = (m + 2)**2
    I = zeros(((2*m + 3)**2, N))
    A = zeros((2*m + 3, m + 2))
    B = zeros((2*m + 3, 2*m + 4))
    for i in range(1, 2*m + 2):
        n = int(i/2)
        if i % 2 == 0:
            A[i, n] = 4.0
            B[i, (n, n + m + 2)] = 2.0, 2.0
        else:
            A[i, (n, n + 1)] = 2.0, 2.0
            B[i,(n, n + 1, n + m + 2, n + m + 3)] = 1.0, 1.0, 1.0, 1.0
    for i in range(1, 2*m + 2):
        if i % 2 == 0:
            n = int(i/2)
            I[i*(2*m + 3):(2*m + 3)*(i + 1), n*(m + 2):(n + 1)*(m + 2)] = A
        else:
            n = int(i/2)
            I[i*(2*m + 3):(2*m + 3)*(i + 1), n*(m + 2):n*(m + 2) + 2*m + 4] = B
    U = np.dot(I, U) / 4.0
    for k in range(0, len(F)):
        if F[k] != 0:
            U[k] = F[k]
    return U


def restrict(U, m, F = zeros((1,1))):
    n = int((m - 1) / 2)
    N = (m + 2) ** 2
    A = np.zeros((n + 2, 3 * (m + 2)))
    I = np.zeros(((n + 2) ** 2, N))
    for i in range(0, n):
        A[i + 1, (2 * i + 1, 2 * i + 2, 2 * i + 3, 2 * i + m + 3,
                  2 * i + m + 4, 2 * i + m + 5, 2 * i + 2 * (m + 2) + 1, 2 * i +
                  2 * (m + 2) + 2, 2 * i + 2 * (m + 2) + 3)] = 1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0
    for i in range(0, n):
        I[(i + 1) * (n + 2):(i + 2) * (n + 2), (2 * i + 1) * (m + 2):(2 * (i + 1) + 2) * (m + 2)] = A
    U = np.dot(I, U) / 16.0
    for k in range(0, len(F)):
        if F[k] != 0:
            U[k] = F[k]
    return U

def vcycle(m, U, A, F, eta1 = 3, eta2 = 3, numcycles = 5, cyclenum = 0, vlist = []):
    if cyclenum == 0:
        vlist.append(F)
    eh = np.zeros((2*m + 1, 1))
    f = lambda x, y: x*0
    if cyclenum < numcycles:
        vh = gs(U, A, F, (m + 2) ** 2, maxiters=eta1)
        vlist.append(vh)
        rh = F - np.dot(A, vh)
        rh = restrict(rh, m)
        m = int((m - 1) / 2)
        eh = np.zeros(((m + 2)**2, 1))
        A, U, F, P, X = Poisson2D(m, f, bvals = True)
        cyclenum += 1
        eh = vcycle(m, eh, A, rh, eta1 = eta1, eta2 = eta2, numcycles = numcycles, cyclenum = cyclenum, vlist = vlist)
    vh = np.linalg.solve(A, F)
    vlist.reverse()
    for i in range(0, numcycles):
        vh = eh + interpolate(vh, m)
        m = 2 * m + 1
    A, U, F, P, X = Poisson2D(m, f, bvals = True)
    vh = gs(vh, A, vlist[-1], (m + 2)**2, maxiters = eta2)
    return vh

def fmg(m, fh, f, eta0 = 1, eta1 = 3, eta2 = 3, numcycles = 5, cyclenum = 0, flist = []):

    vh = np.zeros(((m + 2)**2, 1))
    if cyclenum < numcycles:
            fh = restrict(fh, m)
            cyclenum += 1
            m = int((m - 1) / 2)
            vh = fmg(m, fh, f, eta0=eta0, eta1=eta1, eta2=eta2, numcycles=numcycles, cyclenum=cyclenum)
            m = 2 * m + 1
    A, U, F, P, X = Poisson2D(2*m + 1, f, bvals=True)
    vh = interpolate(vh, m, F)
    for i in range(eta0):
        vh = vcycle(2*m + 1, vh, A, F, eta1 = eta1, eta2 = eta2, numcycles = numcycles - cyclenum + 1)
    return vh

