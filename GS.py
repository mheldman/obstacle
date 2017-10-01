import numpy as np

def gs(U, A, F, m, maxiters = 3, tol = 10**-8, P = U*-np.inf):
    for j in range(0,maxiters):
        r = F - np.dot(A,U)
        if np.linalg.norm(r, np.inf) < tol:
            break
        for i in range(0,m):
            U[i] = max(P[i], (1/A[i,i])*(F[i] - np.dot(A[i , :],U) + A[i, i]*U[i]))
    return U