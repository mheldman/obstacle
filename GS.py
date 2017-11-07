import numpy as np

def gs(U, A, F, m, maxiters = 3, P = False):
    if P == False:
        for j in range(0,maxiters):
            for i in range(0,m):
                U[i] = (1/A[i,i])*(F[i] - A[i, :].dot(U) + A[i, i] * U[i])
        print("Maximum iterations exceeded.")
        return U
    else:
        for j in range(0, maxiters):
            for i in range(0, m):
                U[i] = max(0.0, (1 / A[i, i]) * (F[i] - A[i, :].dot(U) + A[i, i] * U[i]))
        return U
