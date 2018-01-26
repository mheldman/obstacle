from numpy import zeros, pi, dot, linspace, sin
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

m = 20
A = zeros((m, m))
for i in range(1, m - 1):
    A[i, (i - 1, i, i + 1)] = 1.0, -2.0, 1.0
A[0, (0, 1)] = -2.0, 1.0
A[m - 1, (m - 2, m - 1)] = 1.0, -2.0
F = zeros(m)
F[0] = -1.0
F[m - 1] = -1.0
h = 6*pi/(m + 1)
F = F/h**2
A = A/h**2
X = linspace(0, 6*pi, m + 2)
plt.plot(X, sin(X) + 1)
U = sin(X[1:m + 1]) + 1

def jacobi(A, U, F, maxiters = 2):
    for i in range(maxiters):
        Uold = zeros(len(U))
        for j in range(len(U)):
            Uold[j] = U[j]
        for j in range(len(U)):
            U[j] = (1/A[j, j])*(F[j] - dot(A[j, :], Uold)) + Uold[j]
    return U

U = jacobi(A, U, F)
Unew = zeros(m + 2)
Unew[1:m + 1] = U
Unew[0] = 1
Unew[m + 1] = 1

plt.plot(X, Unew)




