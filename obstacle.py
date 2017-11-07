import numpy as np
from GS import gs
from ReducedSpace import reducedspace
from Poisson2D import Poisson2D
from cvxopt import solvers, matrix
from blockprint import*

def obstacleqp(psi, m, f, g, a = -1.0, b = 1.0):
  A, U, P, F, X = Poisson2D(m, f, a = a, b = b, psi = psi, g = g)
  blockPrint()
  N = m**2
  sol = solvers.qp(matrix(-A), matrix(F), matrix(-np.identity(N)), matrix(-P), initvals = matrix(U))
  U = sol['x']
  enablePrint()
  return U

def obstaclersp(psi, m, f, g, a = -1.0, b = 1.0, bvals = True):
    A, U, P, F, X = Poisson2D(m, f, a = a, b = b, psi = psi, g = g, bvals=bvals)
    L = lambda T: -A.dot(T + P) + F
    dL = lambda U: -A
    U = reducedspace(L, dL, U)
    return U + P

def obstaclecpgs(psi, m, f, g, a = -1.0, b = 1.0):
    A, U, P, F, X = Poisson2D(m, f, a = a, b = b, psi = psi, g = g)
    N = m**2
    U = gs(U, A, F, N, maxiters = 1000, tol = 10**-8, P = P)
    return U
