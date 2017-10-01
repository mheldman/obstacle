import numpy as np

def interpolate(U,m,F = None):
    print(m)
    N = (m + 2)**2
    I = np.zeros(((2*m + 3)**2,N))
    A = np.zeros((2*m + 3,m + 2))
    B = np.zeros((2*m + 3,2*m + 4))
    for i in range(0,2*m + 3):
        n = int(i/2)
        if i == 0:
            pass
        elif i == 2*m + 2:
            pass
        elif i%2 == 0:
            A[i,n] = 4.0
            B[i,(n,n + m + 2)] = 2.0, 2.0
        else:
            A[i,(n,n + 1)] = 2.0, 2.0
            B[i,(n,n + 1,n + m + 2,n + m + 3)] = 1.0, 1.0, 1.0, 1.0
    for i in range(1,2*m + 2):
        if i%2 == 0:
            n = int(i/2)
            I[i*(2*m + 3):(2*m + 3)*(i + 1),n*(m + 2):(n + 1)*(m + 2)] = A
        else:
            n = int(i/2)
            I[i*(2*m + 3):(2*m + 3)*(i + 1),n*(m + 2):n*(m + 2) + 2*m + 4] = B
    U = np.dot(I,U)/4.0
    if F != None:
        kk = lambda i,j: j * (2*m + 3) + i
        k = 0
        for i in range(0, 2*m + 3):
            for j in range(0, 2*m + 3):
                if i == 0 or i == 2*m + 2 or j%(2*m + 2) == 0 or j%(2*m + 3) == 0:
                    U[kk(j,i)] = F2[k]
                    k += 1
    return U