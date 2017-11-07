import numpy as np
from numpy import zeros
from scipy import sparse
from scipy.sparse.linalg import spsolve
from GS import gs
from Poisson2D import Poisson2D
from obstacle import obstaclersp

def interpolate(U, m, F = zeros((1,1))):
    N = (m + 2)**2
    I = sparse.lil_matrix(((2*m + 3)**2, N))
    A = np.zeros((2*m + 3, m + 2))
    B = np.zeros((2*m + 3, 2*m + 4))
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
    I = I.tocsr()
    U = I.dot(U) / 4.0
    for k in range(0, len(F)):
        if F[k] != 0:
            U[k] = F[k]
    return U


def restrict(U, m, F = zeros((1,1))):
    n = int((m - 1) / 2)
    N = (m + 2) ** 2
    A = zeros((n + 2, 3 * (m + 2)))
    I = sparse.lil_matrix(((n + 2) ** 2, N))
    for i in range(0, n):
        A[i + 1, (2 * i + 1, 2 * i + 2, 2 * i + 3, 2 * i + m + 3,
                  2 * i + m + 4, 2 * i + m + 5, 2 * i + 2 * (m + 2) + 1, 2 * i +
                  2 * (m + 2) + 2, 2 * i + 2 * (m + 2) + 3)] = 1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0
    for i in range(0, n):
        I[(i + 1) * (n + 2):(i + 2) * (n + 2), (2 * i + 1) * (m + 2):(2 * (i + 1) + 2) * (m + 2)] = A
    I = I.tocsr()
    U = I.dot(U) / 16.0
    for k in range(0, len(F)):
        if F[k] != 0:
            U[k] = F[k]
    return U

def sweep(vh, fh, m, eta, P = False):
    A, _, _, _, _ = Poisson2D(m, bvals=True)
    N = (m + 2) ** 2
    vh = gs(vh, A, fh, N, maxiters = eta, P=P)
    return vh

def vcycle(m, vh, A, fh, eta1 = 3, eta2 = 3, numcycles = 5, cyclenum = 0):
    if cyclenum < numcycles:
        vh = sweep(vh, fh, m, eta1)
        f2h = restrict(fh - A.dot(vh), m)
        m = int((m - 1) / 2)
        N = (m + 2) ** 2
        v2h = np.zeros(N)
        cyclenum += 1
        A, _, _, _, _ = Poisson2D(m, bvals=True)
        v2h = vcycle(m, v2h, A, f2h, eta1 = 3, eta2 = 3, numcycles = numcycles, cyclenum = cyclenum)
    else:
        vh = spsolve(A, fh)
        return vh
    vh = vh + interpolate(v2h, m)
    m = 2 * m + 1
    vh = sweep(vh, fh, m, eta2)
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
        vh = zeros((m + 2) ** 2)
        return vh
    A, _, _, _, _ = Poisson2D(m, bvals=True)
    vh = vcycle(m, vh, A, fh, eta1 = eta1, eta2 = eta2, numcycles = numcycles - cyclenum + 1)
    return vh

def pfas(m, uh, Ah, fh, eta=1, numcycles = 5, cyclenum = 0):
    if cyclenum < numcycles:
            vh = gs(uh, Ah, fh, (m + 2) ** 2, maxiters=eta, P=True)
            r2h = restrict(fh - Ah.dot(vh), m)
            v2h = restrict(vh, m)
            cyclenum += 1
            m = int((m - 1) / 2)
            A2h, _, _, _, _ = Poisson2D(m, bvals=True, a=-2.0, b=2.0)
            v2h = np.maximum(v2h, 0.0)
            f2h = r2h + A2h.dot(v2h)
            u2h = pfas(m, v2h, A2h, f2h, eta=eta, numcycles=numcycles, cyclenum=cyclenum)
            e2h = u2h - v2h
    else:
        du = 1.0
        uold = np.zeros(len(uh))
        while du > 10**-5:
            for i in range(0, len(uh)):
                uold[i] = uh[i]
            uh = gs(uh, Ah, fh, (m + 2)**2, maxiters=eta, P=True)
            du = m*np.linalg.norm(uold - uh)
            print(du)
        return uh
    vh = vh + interpolate(e2h, m)
    m = 2 * m + 1
    vh = np.maximum(vh, 0.0)
    vh = gs(vh, Ah, fh, (m + 2) ** 2, maxiters=eta, P=True)
    return vh