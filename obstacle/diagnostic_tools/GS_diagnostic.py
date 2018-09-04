import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
from time import time
from pyamg.relaxation.relaxation import gauss_seidel
from pyamg.relaxation.relaxation import projected_gauss_seidel
#from pyamg import projected_gauss_seidel

def gs(A, x, b, maxiters = 1):

    gauss_seidel(A, x, b, iterations=maxiters)
    #L = scipy.sparse.tril(A, 0, format='csr')
    #U = scipy.sparse.triu(A, 1, format='csr')
    #c = b - U.dot(x)
    #x -= x
    #for j in range(0, maxiters):
    #    x += scipy.sparse.linalg.spsolve_triangular(L, c, lower=True) #The details of the backward/forward substitution depend in great
        # deal on the (parallel) data structure used to store the lower and upper triangular part. This is why
        # the lower/upper triangular solvers are tied to the same code that does the factorization;
        # for example I cannot factor with MUMPS and then do the triangular solves
        # with SuperLU_Dist since they assume different and complicated (parallel) data structures.

    '''
    m = len(x)
    for j in range(0, maxiters):
        for i in range(0, m):
            x[i] = (1/A[i,i])*(b[i] - A[i, :].dot(x) + A[i, i] * x[i])
    return x
    '''


def pgs(A, x, b, maxiters = 1):#, plot_active=False, bounds = None, mx=None, my=None):
    projected_gauss_seidel(A, x, b, iterations=maxiters)
    #L = scipy.sparse.tril(A, 0, format='csr')
    #U = scipy.sparse.triu(A, 1, format='csr')
    #c = b - U.dot(x)
    #x -= x #find different workaround for in place editing
    #for j in range(0, maxiters):
     #   x += scipy.sparse.linalg.spsolve_triangular_project(L, c, lower=True)
    '''
    if plot_active:
        x1, x2, y1, y2 = bounds
        kk = lambda i, j: (mx + 2) * i + j
        Z = np.zeros((mx + 2, my + 2))
        for i in range(0, my + 2):
            for j in range(0, mx + 2):
                k = kk(i, j)
                Z[j, i] = U[k]
        X = np.linspace(x1, x2, mx + 2)
        Y = np.linspace(y1, y2, my + 2)
        C, B = np.meshgrid(X, Y)
        C, B = np.transpose(C), np.transpose(B)
        plt.plot(C[[Z < 1e-10]], B[[Z < 1e-10]], 'o', color='k')
        plt.ion()
        plt.xlim(x1, x2)
        plt.ylim(y1, y2)
        plt.show()
        plt.pause(2)
        plt.close('all')
'''