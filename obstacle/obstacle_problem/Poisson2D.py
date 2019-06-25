
from scipy import sparse
from numpy import zeros, linspace, arange
import numpy as np

__all__ = ['poisson2d', 'rhs']

f = lambda x, y: 0.*x
g = f

def poisson2d(mx, my=None, x1=-1.0, x2=1.0, y1=None, y2=None, bvals = True, symmetric=False): #use stencil.py from pyamg
  
    if y1 is None:
        y1 = x1
        y2 = x2

    if my is None:
        my = mx

    hx = (x2 - x1) / (mx + 1)
    hy = (y2 - y1) / (my + 1)
    a, b = 1. / hy**2, 1. / hx**2

    if bvals:
        N = (mx + 2) * (my + 2)
        diag1 = -2. * (a + b) * np.ones(N)
        index_list = range(0, mx + 2) + range((my + 1) * (mx + 2), N)\
                     + range(mx + 2, N - (mx + 2), mx + 2) + range(mx + 1, N, mx + 2)
        diag1[index_list] = 1.0
        diag2 = a*np.ones(N)
        diag4 = diag2[0:N - (mx + 2)]
        index_list = range(0, mx + 2) + range(mx + 2, N - (mx + 2), mx + 2) + range(mx + 1, N - (mx + 2), mx + 2)
        diag4[index_list] = 0.
        index_list = range(mx + 2, N - (mx + 2), mx + 2) + range(mx + 1, N - (mx + 2), mx + 2) + range(N - 2*(mx + 2), N - (mx + 2))
        diag2 = diag2[mx + 2:N]
        diag2[index_list] = 0.
        diag3 = b*np.ones(N)
        diag5 = diag3[1:N]
        diag3 = diag3[0:N-1]
        index_list = (diag1 == 1.)
        diag3[index_list[0:N-1]] = 0.
        diag5[index_list[1:N]] = 0.
        A = sparse.diags([diag2, diag5, diag1, diag3, diag4], [-mx - 2, -1, 0, 1, mx + 2], (N, N), format='csr')
        if symmetric:
          B = A.T - A
          A += B.minimum(0.)
        return A

    else:

        N = mx * my
        A = sparse.diags([a, b, -2. * (a + b), b, a], [-mx, -1, 0, 1, mx], (N, N), format='csr')
        return A


def rhs(f, mx, my=None, x1=-1.0, x2=1.0, y1=None, y2=None, g = g, bvals = True, symmetric = False):

    if y1 is None:
        y1 = x1
        y2 = x2

    if my is None:
        my = mx

    N = (mx + 2) * (my + 2)

    if bvals:
      
        hx = (x2 - x1) / (mx + 1)
        hy = (y2 - y1) / (my + 1)
        a, b = hx / hy, hy / hx
        X = linspace(x1, x2, mx + 2)
        Y = linspace(y1, y2, my + 2)
        F = zeros(N)
        '''
        if type(f(X[0:2], Y[0:2])) in [int, float] or type(g(X[0:2], Y[0:2])) in [int, float]:
            kk = lambda i, j: j * (mx + 2) + i
            for j in range(0, my + 2):
                for i in range(0, mx + 2):
                    k = kk(i, j)
                    if j == 0 or j == my + 1 or i == 0 or i == mx + 1:
                      F[k] = g(X[i], Y[j])
                    else:
                      F[k] = hx * hy * f(X[i], Y[j])
                      if symmetric:
                        if j == 1:
                          F[k] -= a*g(X[i], Y[0])
                        if j == my:
                            F[k] -= a*g(X[i], Y[-1])
                        if i == 1:
                            F[k] -= b*g(X[0], Y[j])
                        if i == mx:
                            F[k] -= b*g(X[-1], Y[j])

        '''
        [X, Y] = np.meshgrid(X, Y)
        X, Y = X.flatten(), Y.flatten()
        bvalarray = (X == x1) | (X == x2) | (Y == y1) | (Y == y2)
        notbvals = ~bvalarray
        F[notbvals] = f(X[notbvals], Y[notbvals])
        F[bvalarray] = g(X[bvalarray], Y[bvalarray])
        if symmetric:
          F[abs(X - x1 - hx) < 1e-15] -= g(X[X == x1], Y[X == x1]) / hx**2
          F[abs(X - x2 + hx) < 1e-15] -= g(X[X == x2], Y[X == x2]) / hx**2
          F[abs(Y - y1 - hy) < 1e-15] -= g(X[Y == y1], Y[Y == y1]) / hy**2
          F[abs(Y - y2 + hy) < 1e-15] -= g(X[Y == y2], Y[Y == y2]) / hy**2



    if not bvals:
        hx = (x2 - x1) / (mx + 1)
        hy = (y2 - y1) / (my + 1)
        a, b = hx / hy, hy / hx
        
        X = linspace(x1 + hx, x2 - hx, mx)
        Y = linspace(y1 + hy, y2 - hy, my)
        [X, Y] = np.meshgrid(X, Y)
        X, Y = X.flatten(), Y.flatten()
        
        F = zeros(mx*my)
        G = g(X, Y)
        
        kk = lambda i, j: j * mx + i
        for j in range(0, my):
            for i in range(0, mx):
                k = kk(i, j)
                F[k] = f(X[i], Y[j])
                if j == 0:
                    G[k] -= a*g(X[i], y1)
                if j == my - 1:
                    G[k] -= a*g(X[i], y2)
                if i == 0:
                    G[k] -= b*g(x1, Y[j])
                if i == mx - 1:
                    G[k] -= b*g(x2, Y[j])
        F = F - G

    return F
