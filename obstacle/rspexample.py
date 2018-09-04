import numpy as np
from ReducedSpace import reducedspace

def F(x):
    A = np.zeros(4)
    A[0] = 3 * x[0] ** 2 + 2 * x[0] * x[1] + 2 * x[1] ** 2 + x[2] + 3 * x[3] - 6
    A[1] = 2 * x[0] ** 2 + x[1] ** 2 + x[0] + 10 * x[2] + 2 * x[3] - 2
    A[2] = 3 * x[0] ** 2 + x[0] * x[1] + 2 * x[1] ** 2 + 2 * x[2] + 9 * x[3] - 9
    A[3] = x[0] ** 2 + 3 * x[1] ** 2 + 2 * x[2] + 3 * x[3] - 3
    return A


def gradF(x):
    A = np.matrix([[6 * x[0] + 2 * x[1], 2 * x[0] + 4 * x[1], 1, 3], \
                   [4 * x[0] + 1, 2 * x[1], 10, 2], \
                   [6 * x[0] + x[1], x[0] + 4 * x[1], 2, 9], \
                   [2 * x[0], 6 * x[1], 2, 3]])
    return A


x0 = np.array([100, -1, 2, 0])
#exact = np.array([1,0,3,0])
exact = np.array([np.sqrt(6) / 2, 0, 0, .5])
xstar, errlist = reducedspace(F, gradF, x0, 10 ** -8, exact)
print(errlist[len(errlist) - 1])