import numpy as np
from scipy.sparse.linalg import cg

def Fomega(A, x):
    n = len(x)
    y = np.zeros((n, 1))
    y[[x > 0]] = A[[x > 0]]
    y[[x <= 0]] = np.minimum(A[[x <= 0]], 0.0)
    return y


def pi(x):
    y = x
    y[[x < 0]] = 0
    return y


def reducedspace(F, gradF, x0, tol = 10**-5, exact = True, sigma=10 ** -4, beta = .5, gamma = 10 ** -12):
    errlist = []
    n = len(x0)
    k = 0
    xk = x0
    A = F(xk)
    FO = Fomega(A, xk)
    pik = pi(x0)
    while np.linalg.norm(FO) > tol and k < 100:  # might use ||[x1*F1, x2*F2, ..., xn*Fn]||_inf
        k += 1
        print(k, xk)
        Axk = []
        Ixk = []
        for i in range(0, n):
            if xk[i] == 0 and A[i] > 0:
                Axk.append(i)
            else:
                Ixk.append(i)
        d = np.zeros((n, 1))
        temp = gradF(xk)
        m = len(Ixk)
        B = np.zeros((m, m))
        for i in range(0, m):
            for j in range(0, m):
                B[i, j] = temp[Ixk[i], Ixk[j]]

        dIxk, info = cg(B, -A[Ixk], tol=10 ** -5)
        j = 0
        for i in Ixk:
            d[i] = dIxk[j]
            j += 1

        alpha = beta
        fail = False
        Ak = F(pik)
        while np.linalg.norm(Fomega(Ak, pik)) > (1 - sigma * alpha) * np.linalg.norm(FO):
            pik = pi(xk + alpha * d)
            Ak = F(pik)
            alpha *= beta
            if alpha < gamma:
                fail = True
                break

        if fail:
            alpha = beta
            d = -F(xk)
            while np.linalg.norm(Fomega(Ak, pik)) > (1 - sigma * alpha) * np.linalg.norm(FO):
                alpha = alpha * beta
                pik = pi(xk + alpha * d)
                Ak = F(pik)
            if beta < gamma:
                print('Could not provide sufficient decrease. Process terminated iteration', k)
                return xk

        xk = pik
        A = F(xk)
        FO = Fomega(A, xk)
    print('\n', 'xk =', xk, '\n', 'F(xk) =', F(xk), '\n', 'F(xk)*xk =', np.dot(np.transpose(F(xk)), xk))
    return xk
