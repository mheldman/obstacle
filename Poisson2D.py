#!/usr/bin/env python3

from scipy import sparse
from numpy import zeros, identity, linspace

f = lambda x, y: 0
g = f

def poisson2d(m, x1=-1.0, x2=1.0, y1=None, y2=None, bvals = True):

    if y1 is None:
        y1 = x1
        y2 = x2

    hx = (x2 - x1) / (m + 1)
    hy = (y2 - y1) / (m + 1)

    if bvals:

        N = (m + 2) ** 2
        A = sparse.lil_matrix((N, N))
        T = zeros((m + 2, m + 2))
        I = zeros((m + 2, m + 2))
        I[1:m + 1, 1:m + 1] = identity(m) / hy ** 2
        for i in range(1, m + 1):
            T[i, (i - 1, i, i + 1)] = 1.0 / hx**2, -2.0 / hx**2 + -2.0 / hy**2, 1.0 / hx**2
        T[0, 0] = 1.0
        T[-1, -1] = 1.0
        for i in range(1, m + 1):
            A[i * (m + 2):(i + 1) * (m + 2), i * (m + 2):(i + 1) * (m + 2)] = T
            A[i * (m + 2):(i + 1) * (m + 2), (i - 1) * (m + 2):i * (m + 2)] = I
            A[i * (m + 2):(i + 1) * (m + 2), (i + 1) * (m + 2):(i + 2) * (m + 2)] = I
        A[0:m + 2, 0:m + 2] = identity(m + 2)
        A[(m + 2) * (m + 1): N, (m + 2) * (m + 1): N] = identity(m + 2)
        A = A.tocsr()
        return A

    else:

        N = m ** 2
        A = sparse.lil_matrix((N, N))
        T = zeros((m, m))
        I = identity(m)
        for i in range(1, m - 1):
            T[i, (i - 1, i, i + 1)] = 1.0 / hx**2, -2.0 / hx**2 + -2.0 / hy**2, 1.0 / hy**2
        T[0, (0, 1)] = -2.0 / hx**2 + -2.0 / hy**2, 1.0 / hx**2
        T[-1, (-2, -1)] = 1.0 / hx**2, -2.0 / hx**2 + -2.0 / hy**2
        for i in range(1, m - 1):
            A[i * m:(i + 1) * m, i * m:(i + 1) * m] = T
            A[i * m:(i + 1) * m, (i + 1) * m:m * (i + 2)] = I
            A[i * m:(i + 1) * m, m * (i - 1):i * m] = I
        A[0:m, 0:m] = T
        A[0:m, m:2 * m] = I
        A[m * (m - 1):N, m * (m - 2):m * (m - 1)] = I
        A[m * (m - 1):N, m * (m - 1):N] = T
        A = A.tocsr()
        return A


def rhs(m, f, x1=-1.0, x2=1.0, y1=None, y2=None, g = g, bvals = True):

    if y1 is None:
        y1 = x1
        y2 = x2

    X = linspace(x1, x2, m + 2)
    Y = linspace(y1, y2, m + 2)
    N = (m + 2)**2
    F = zeros(N)

    if bvals:

        kk = lambda i, j: j * (m + 2) + i
        for j in range(0, m + 2):
            for i in range(0, m + 2):
                k = kk(i, j)
                if j == 0 or j == m + 1 or i == 0 or i == m + 1:
                    F[k] = g(X[i], Y[j])
                else:
                    F[k] = f(X[i], Y[j])

    else:

        kk = lambda i, j: j * m + i
        for j in range(0, m):
            for i in range(0, m):
                k = kk(i, j)
                F[k] = f(X[i], Y[j])
                if j == 0:
                    G[k] += g(X[i], y1) / hy ** 2
                if j == m - 1:
                    G[k] += g(X[i], y2) / hy ** 2
                if i == 0:
                    G[k] += g(x1, Y[j]) / hx ** 2
                if (i + 1) % m == 0:
                    G[k] += g(x2, Y[j]) / hx ** 2
        F = F - G

    return F