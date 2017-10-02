import numpy as np
from numpy import zeros

def interpolate(U, m, F = zeros(1)):
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
    U = np.dot(I,U) / 4.0
    U[[F[k] != 0]] = F[[F[k] != 0]]
    return U


def restrict(U, m, F = zeros(1)):
    n = np.floor(m / 2)
    n = int(n)
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
    k = 0
    U[[F[k] != 0]] = F[[F[k] != 0]]
    return U

def vcycle(m, U, A, F, minm = 25, eta1 = 3, eta2 = 3, numcycles = 5,cyclenum = 0, v = None):
    vh = gs(U, A, F, m, maxiters = eta1)
    print(vh.shape)
    print(m,cyclenum)
    if cyclenum < numcycles:
        rh = F - np.dot(A,vh)
        rh = restrict(rh,m)
        m = int(np.floor(m/2))
        eh = np.zeros(((m+2)**2,1))
        cyclenum += 1
        A, U, F, X = obstacle2D(m,f)
        eh = vcycle(m, eh, A, rh, cyclenum = cyclenum, numcycles = numcycles, v = vh)
        return eh
    vh = v + interpolate(vh,m)
    vh = gs(U, A, F, m, maxiters = eta2)
    return vh

