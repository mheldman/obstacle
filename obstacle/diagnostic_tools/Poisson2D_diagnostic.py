
from scipy import sparse
from numpy import zeros, identity, linspace
import numpy as np
import matplotlib.pyplot as plt
from time import time

f = lambda x, y: 0.0*x
g = f

def poisson2d(mx, my=None, x1=-1.0, x2=1.0, y1=None, y2=None, bvals = True): #use stencil.py from pyamg

    if y1 is None:
        y1 = x1
        y2 = x2

    if my is None:
        my = mx

    hx = (x2 - x1) / (mx + 1)
    hy = (y2 - y1) / (my + 1)
    a, b = 1. / hy ** 2, 1. / hx ** 2

    if bvals:
        N = (mx + 2) * (my + 2)
        diag1 = -2. * (a + b) * np.ones(N)
        index_list = range(0, mx + 2) + range((my + 1) * (mx + 2), N)\
                     + range(mx + 2, N - (mx + 2), mx + 2) + range(mx + 1, N, mx + 2)
        diag1[index_list] = 1.0
        diag2 = np.ones(N) / hy ** 2
        diag4 = diag2[0:N - (mx + 2)]
        index_list = range(0, mx + 2) + range(mx + 2, N - (mx + 2), mx + 2) + range(mx + 1, N - (mx + 2), mx + 2)
        diag4[index_list] = 0.0
        index_list = range(mx + 2, N - (mx + 2), mx + 2) + range(mx + 1, N - (mx + 2), mx + 2) + range(N - 2*(mx + 2), N - (mx + 2))
        diag2 = diag2[mx + 2:N]
        diag2[index_list] = 0.0
        diag3 = np.ones(N) / hx ** 2
        diag5 = diag3[1:N]
        diag3 = diag3[0:N-1]
        index_list = (diag1 == 1.)
        diag3[index_list[0:N-1]] = 0.
        diag5[index_list[1:N]] = 0.
        A = sparse.diags([diag2, diag5, diag1, diag3, diag4], [-mx - 2, -1, 0, 1, mx + 2], (N, N), format='csr')
        return A

    else:

        N = mx * my
        A = sparse.diags([a, b, -2. * (a + b), b, a], [-mx, -1, 0, 1, mx], (N, N), format='csr')
        return A


def rhs(f, mx, my=None, x1=-1.0, x2=1.0, y1=None, y2=None, g = g, bvals = True):

    if y1 is None:
        y1 = x1
        y2 = x2

    if my is None:
        my = mx

    N = (mx + 2) * (my + 2)

    if bvals:

        X = linspace(x1, x2, mx + 2)
        Y = linspace(y1, y2, my + 2)
        if type(f(X[0:2], Y[0:2])) in [int, float]:
            F = zeros(N)
            kk = lambda i, j: j * (mx + 2) + i
            for j in range(0, my + 2):
                for i in range(0, mx + 2):
                    k = kk(i, j)
                    if j == 0 or j == my + 1 or i == 0 or i == mx + 1:
                        F[k] = g(X[i], Y[j])
                    else:
                        F[k] = f(X[i], Y[j])

        else:
            [X, Y] = np.meshgrid(X, Y)
            X, Y = X.flatten(), Y.flatten()
            F = f(X, Y)
            F[(X == x1) | (X == x2) | (Y == y1) | (Y == y2)] = g(X, Y)[(X == x1) | (X == x2) | (Y == y1) | (Y == y2)]

    if not bvals:

        kk = lambda i, j: j * mx + i
        for j in range(0, my):
            for i in range(0, mx):
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
