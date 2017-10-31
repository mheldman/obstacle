#!/usr/bin/env python3

'''
Discretizes the 2D Poission problem u'' = f on the square region Omega = [a,b]x[a,b].
Required inputs are m, the number of unknowns on a 1D slice of the region, and f, which should be a real-valued function defined on Omega.
Optional inputs are a and b, which define Omega, g, which defines the boundary values, and psi, which defines the obstacle if solving an obstacle problem.
The grid spacing in x and y is (b - a)/(m + 1). The discreization defined here constructs the system of equations U[i] = F[i] for points (x, y) on the boundary of Omega (i.e. x = pm 1 or y = pm 1), in which case F[i] is defined to be g(x, y), and (U[i - 1] + U[i + 1] + U[i + m + 2] + U[i - m - 2\ - 4U[i])/h^2 if U[i] does not represent a boundary point.
Note that this function sets up, but does not solve, the system of equations.
'''

import numpy as np
from scipy import sparse
psi = lambda x, y: -np.inf
g = lambda x, y: 0
f = lambda x, y: 0


def Poisson2D(m, f = f, a=-1.0, b=1.0, psi = psi, g = g, bvals = False):
    if not bvals: #specifies whether or not to explicitly include boundary values in the Poisson discretization
        N = m ** 2
        h = (b - a) / (m + 1)
        X = np.linspace(a + h, b - h, m)
        A = sparse.csr_matrix((N, N))
        F = sparse.csr_matrix((N, 1))
        G = sparse.csr_matrix((N, 1))
        U = sparse.csr_matrix((N, 1))
        P = sparse.csr_matrix((N, 1))
        T = sparse.csr_matrix((m, m))
        I = sparse.diags(np.ones(m), [0])
        for i in range(1, m - 1):
            T[i, (i - 1, i, i + 1)] = 1.0, -4.0, 1.0
        T[0, (0, 1)] = -4.0, 1.0
        T[-1, (-2, -1)] = 1.0, -4.0
        for i in range(1, m - 1):
            A[i * m:(i + 1) * m, i * m:(i + 1) * m] = T
            A[i * m:(i + 1) * m, (i + 1) * m:m * (i + 2)] = I
            A[i * m:(i + 1) * m, m * (i - 1):i * m] = I
        A[0:m, 0:m] = T
        A[0:m, m:2 * m] = I
        A[m * (m - 1):N, m * (m - 2):m * (m - 1)] = I
        A[m * (m - 1):N, m * (m - 1):N] = T
        A = A/h**2
        kk = lambda i, j: j * m + i
        for j in range(0, m):
            for i in range(0, m):
                k = kk(i, j)
                F[k] = f(X[i], X[j])
                P[k] = psi(X[i], X[j])
                U[k] = max(P[k], 0.0)
                if j == 0:
                    G[k] += g(X[i], a)/h**2
                if j == m - 1:
                    G[k] += g(X[i], b)/h**2
                if i % m == 0:
                    G[k] += g(a, X[j])/h**2
                if (i + 1) % m == 0:
                    G[k] += g(b, X[j])/h**2
        F = F - G
        return A, U, P, F, X

    else:
        N = (m + 2) ** 2
        h = (b - a) / (m + 1)
        X = np.linspace(a, b, m + 2)
        A = sparse.lil_matrix((N, N))
        F = np.zeros(N)
        P = np.zeros(N)
        U = np.zeros(N)
        T = np.zeros((m + 2, m + 2))
        I = np.zeros((m + 2, m + 2))
        I[1:m + 1, 1:m + 1] = np.identity(m) / h ** 2
        for i in range(1, m + 1):
            T[i, (i - 1, i, i + 1)] = 1.0, -4.0, 1.0
        T = T / h ** 2
        T[0, 0] = 1.0
        T[-1, -1] = 1.0
        for i in range(1, m + 1):
            A[i * (m + 2):(i + 1) * (m + 2), i * (m + 2):(i + 1) * (m + 2)] = T
            A[i * (m + 2):(i + 1) * (m + 2), (i - 1) * (m + 2):i * (m + 2)] = I
            A[i * (m + 2):(i + 1) * (m + 2), (i + 1) * (m + 2):(i + 2) * (m + 2)] = I
        A[0:m + 2, 0:m + 2] = np.identity(m + 2)
        A[(m + 2) * (m + 1): N, (m + 2) * (m + 1): N] = np.identity(m + 2)
        kk = lambda i, j: j * (m + 2) + i
        for j in range(0, m + 2):
            for i in range(0, m + 2):
                k = kk(i, j)
                if j == 0 or j == m + 1 or i % m == 0 or i % (m + 1) == 0:
                    F[k] = g(X[i], X[j])
                else:
                    F[k] = f(X[i], X[j])
                P[k] = psi(X[i], X[j])
                U[k] = max(P[k], 0.0) #If psi is not defined by the user, U[k] defaults to the zero vector
        A = A.tocsr()
        return A, U, F, P, X
