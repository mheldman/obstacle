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

def sweep(vh, fh, m, eta):
    A, _, _, _, _ = Poisson2D(m, bvals=True)
    N = (m + 2) ** 2
    vh = gs(vh, A, fh, N, maxiters = eta)
    #rh = F - np.dot(A, U)
    return vh #rh, A

def vcycle(m, vh, A, fh, eta1 = 3, eta2 = 3, numcycles = 5, cyclenum = 0):
    if cyclenum < numcycles:
        vh = sweep(vh, fh, m, eta1)
        f2h = restrict(fh - np.dot(A, vh), m)
        m = int((m - 1) / 2)
        N = (m + 2) ** 2
        v2h = np.zeros((N,1))
        cyclenum += 1
        A, _, _, _, _ = Poisson2D(m, bvals=True)
        v2h = vcycle(m, v2h, A, f2h, eta1 = 3, eta2 = 3, numcycles = numcycles, cyclenum = cyclenum)
    else:
        vh = np.linalg.solve(A, fh)
        return vh
    vh = vh + interpolate(v2h, m)
    m = 2*m + 1
    N = (m+2)**2
    A, _, _, _, _ = Poisson2D(m, bvals=True)
    vh = gs(vh, A, fh, N, maxiters=eta2)
    return vh


def fmg(m, fh, eta0 = 1, eta1 = 3, eta2 = 3, numcycles = 5, cyclenum = 0):
    if cyclenum < numcycles:
            f2h = restrict(fh, m)
            cyclenum += 1
            m = int((m - 1) / 2)
            v2h = fmg(m, f2h, eta0=eta0, eta1=eta1, eta2=eta2, numcycles=numcycles, cyclenum=cyclenum)
            vh = interpolate(v2h, m)
            m = 2 * m + 1
    else:
        vh = np.zeros(((m + 2) ** 2, 1))
    A, _, _, _, _ = Poisson2D(m, bvals=True)
    vh = vcycle(m, vh, A, fh, eta1 = eta1, eta2 = eta2, numcycles = numcycles - cyclenum)
    return vh

