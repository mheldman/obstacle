import numpy as np
from numpy import zeros
from scipy import sparse

def interpolate(m):
    N = (m + 2)**2
    I = sparse.lil_matrix(((2*m + 3)**2, N))
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
    I = I.tocsr()
    return I / 4.0

def restrict_fw(m):
    '''
    n = int((m - 1) / 2)
    I = np.transpose(interpolate(n)) / 4.0
    return I
    '''
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
    return I / 16.0


def restrict_inj(m):
    n = int((m - 1) / 2)
    N = (m + 2) ** 2
    I = sparse.lil_matrix(((n + 2) ** 2, N))
    for j in range(n + 2):
        for i in range(n + 2):
            I[i + j * (n + 2), 2 * i + 2 * j * (m + 2)] = 1.0
    I = I.tocsr()
    return I