import numpy as np
from ReducedSpace import reducedspace


def F(x):
    A = np.zeros(2)
    A[0] = x[0] + 9
    A[1] = x[1] / 9 - 10
    return A


def gradF(x):
    A = np.matrix([[1, 0],
                   [0, 1 / 9]])
    return A


x0 = np.array([10000, 100000])
exact = np.array([0.0, 90.0])
xstar, errlist = reducedspace(F, gradF, x0, 10 ** -8, exact)