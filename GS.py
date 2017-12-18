def gs(A, U, F, maxiters = 1):
    m = len(U)
    for j in range(0, maxiters):
        for i in range(0, m):
            U[i] = (1/A[i,i])*(F[i] - A[i, :].dot(U) + A[i, i] * U[i])
    return U

def pgs(A, U, F, maxiters = 1):
    m = len(U)
    for j in range(0, maxiters):
        for i in range(0, m):
            U[i] = max(0.0, (1/A[i, i])*(F[i] - A[i, :].dot(U) + A[i, i] * U[i]))
    return U