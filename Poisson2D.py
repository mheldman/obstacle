
from scipy import sparse
from numpy import zeros, identity, linspace

f = lambda x, y: 0
g = f

def poisson2d(mx, my=None, x1=-1.0, x2=1.0, y1=None, y2=None, bvals = True):

    if y1 is None:
        y1 = x1
        y2 = x2

    if my is None:
        my = mx

    hx = (x2 - x1) / (mx + 1)
    hy = (y2 - y1) / (my + 1)

    if bvals:

        N = (mx + 2) * (my + 2)
        A = sparse.lil_matrix((N, N))
        T = zeros((mx + 2, mx + 2))
        I = zeros((mx + 2, mx + 2))
        I[1:mx + 1, 1:mx + 1] = identity(mx) / hy ** 2
        for i in range(1, mx + 1):
            T[i, (i - 1, i, i + 1)] = 1.0 / hx**2, -2.0 / hx**2 + -2.0 / hy**2, 1.0 / hx**2
        T[0, 0] = 1.0
        T[-1, -1] = 1.0
        for i in range(1, my + 1):
            A[i * (mx + 2):(i + 1) * (mx + 2), i * (mx + 2):(i + 1) * (mx + 2)] = T
            A[i * (mx + 2):(i + 1) * (mx + 2), (i - 1) * (mx + 2):i * (mx + 2)] = I
            A[i * (mx + 2):(i + 1) * (mx + 2), (i + 1) * (mx + 2):(i + 2) * (mx + 2)] = I
        A[0:mx + 2, 0:mx + 2] = identity(mx + 2)
        A[(my + 1)*(mx + 2):N, (my + 1)*(mx + 2):N] = identity(mx + 2)
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


def rhs(f, mx, my=None, x1=-1.0, x2=1.0, y1=None, y2=None, g = g, bvals = True):

    if y1 is None:
        y1 = x1
        y2 = x2

    if my is None:
        my = mx

    X = linspace(x1, x2, mx + 2)
    Y = linspace(y1, y2, my + 2)
    N = (mx + 2) * (my + 2)
    F = zeros(N)

    if bvals:

        kk = lambda i, j: j * (mx + 2) + i
        for j in range(0, my + 2):
            for i in range(0, mx + 2):
                k = kk(i, j)
                if j == 0 or j == my + 1 or i == 0 or i == mx + 1:
                    F[k] = g(X[i], Y[j])
                else:
                    F[k] = f(X[i], Y[j])

    else:

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

