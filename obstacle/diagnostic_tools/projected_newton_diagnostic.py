import numpy as np
from scipy.sparse.linalg import spsolve, inv
import pyamg
from time import time
import matplotlib.pyplot as plt
from GS_diagnostic import pgs
import scipy

'''
Projected Newton algorithm for linear complementarity problems (LCPs).

The reduced space method is designed to solve the nonlinear complementarity problem (NCP):

    Given F:R^n to R^n, find x in R^n satisfying

        F(x) >= 0;
           x >= 0;
    x^T[F(x)] = 0.

For LCPs, f(x) = b - Ax is affine, and the problem reduces to:

    Given A in R^(n x n), b in R^n, find x in R^n satisfying

        b - Ax >= 0;
             x >= 0;
    x^T(b - Ax) = 0.

Note that A is square.

The algorithm uses a modified Newton's method. On each iteration, an active set of indices is computed, defined by

    Active(x) = {i in {1,...,n} : x_i = 0 and [A(x)]_i > -b_i}

and an inactive set

    I(x) = {1,...,n} \ Active(x) = {i in {1,...,n} : x_i > 0 or [A(x)]_i <= 0} (where \ is the set difference operator).

The active set encodes the indices where the nonnegativity constraints on the variables are active, so that 
the function value can be ignored. Once these sets are computed, the reduced Newton's method is applied; the gradient
grad(Ax - b) = A at the current iterate xk is computed, and a next search direction is found by taking a Newton step in 
the space of inactive constraints by approximately solving:

        [Axk]_{I(xk), I(xk)}d = -(Axk - b)_{I(xk)}(xk)

and stepping in the reduced space, i.e.

        xk_{I(xk), I(xk)} = xk_{I(xk), I(xk)} + alpha*d.        

Reference:

S.J. Benson, T.S. Munson, Flexible complementarity solvers for large-scale applications, 
Optim. Methods Softw. 21 (1) (2006) 155168.
 '''


def pi(x):
    y = np.zeros_like(x)
    y[x > 0.0] = x[x > 0.0]
    return y


class PNmethod_lcp_solver:
    '''
    parameters:

    F {tuple, callable}

    '''

    def __init__(self, A, b, tol, maxiters, diagnostics=(), bvals=[]):

        self.A = A
        self.b = b
        self.tol = tol
        self.maxiters = maxiters
        self.num_iterations = 0
        self.xk = np.zeros_like(b)
        self.iterate = None
        self.iterates = []
        self.diagnostics = diagnostics
        self.bvals = bvals

    def compute_Fomega(self, x=None):

        if x is None:

            x, F = self.xk, self.iterate.F
            Fomega = np.minimum(F, 0.0)
            Fomega[x > 0] = F[x > 0]
            self.iterate.Fomega = Fomega
            return Fomega

        else:

            F = self.b - self.A.dot(x)
            Fomega = np.minimum(F, 0.0)
            Fomega[x > 0] = F[x > 0]
            return Fomega

    def plot_active_set(self, u, bounds, mx, my):

        x1, x2, y1, y2 = bounds
        kk = lambda i, j: (mx + 2) * i + j
        Z = np.zeros((mx + 2, my + 2))
        for i in range(0, my + 2):
            for j in range(0, mx + 2):
                k = kk(i, j)
                Z[j, i] = u[k]
        X = np.linspace(x1, x2, mx + 2)
        Y = np.linspace(y1, y2, my + 2)
        A, B = np.meshgrid(X, Y)
        A, B = np.transpose(A), np.transpose(B)
        plt.plot(A[[Z < 1e-10]], B[[Z < 1e-10]], 'o', color='k')
        plt.ion()
        plt.xlim(x1, x2)
        plt.ylim(y1, y2)
        plt.show()
        plt.pause(2)
        plt.close('all')

    def solve(self, init_iterate=None, line_search_params=None, linear_solver='amg', mx=None, my=None, bounds=None):
        print('Linear solver', linear_solver)
        A = self.A
        b = self.b
        if line_search_params is None:
            beta = .5
            sigma = 1e-4
            gamma = 1e-100
        else:
            beta, sigma, gamma = line_search_params

        if init_iterate is None:
            self.iterate = PN_lcp_iterate(pi(spsolve(A, b)), A, b, bvals=self.bvals)
            self.xk = self.iterate.xk
            xk = self.xk
        else:
            self.iterate = PN_lcp_iterate(init_iterate, A, b, bvals=self.bvals)
            self.xk = self.iterate.xk
            xk = self.xk

        residuals = [np.linalg.norm(self.iterate.Fomega, np.inf)]
        print('iteration:', str(self.num_iterations))
        print('residual:', residuals[0], '\n')
        if 'show reduced space' in self.diagnostics:
            self.plot_active_set(xk, bounds, mx, my)

        while residuals[-1] > self.tol and self.num_iterations < self.maxiters:
            '''
            for i in range(2):
                self.iterate.xk = pgs(A, xk, b) #PGS acclerated PN -- finds active set more quickly
                xk = self.iterate.xk
            if linear_solver == 'splu':
                d = spsolve(gradF_active, -F_active)
                temp = spsolve(gradF_active, -b[self.iterate.Ik])
                tempx = xk[self.iterate.Ik]
                for i in range(len(tempx)):
                    print(temp[i] - tempx[i], d[i])
            elif linear_solver == 'amg':
                d = pyamg.solve(gradF_active, -F_active, verb=True, tol=1e-6 / len(xk))
            
            self.iterate.linear_solver_time = time() - tstart
            self.iterate.search_dir = np.zeros_like(xk)
        d2 = self.iterate.xk
        d2[self.iterate.Ik] = temp
             '''
            d = self.iterate.search_dir
            alpha, fail = 1, 0
            tstart = time()
            def f(x):
                 return np.dot(b, x) - (1/2)*np.dot(x, A.dot(x))

            while f(xk) - alpha*np.dot(self.iterate.F, d) >= f(pi(xk - alpha*d)): #Armijo line search
            #while np.linalg.norm(self.compute_Fomega(pi(xk - alpha * d))) \
            #            > (1 - sigma * alpha) * np.linalg.norm(self.iterate.Fomega):         #LCP merit function line search

                alpha *= beta
                if alpha < gamma:
                    fail += 1
                    self.iterate.search_dir = -self.iterate.F
                    d = self.iterate.search_dir
                    alpha = 1
                    print('no sufficient decrease in Newton direction, switching to steepest descent')

                if fail == 2:
                    print('line search failure')
                    break

            print(f(pi(xk - alpha*d)))
            self.iterate.line_search_time = time() - tstart
            self.iterate.alpha = alpha
            self.iterate.line_search_fail = fail
            '''
            for i in range(len(self.iterate.Ik)):
                print(temp[i] - tempx[i], d[self.iterate.Ik][i])
                print(xk[self.iterate.Ik][i] + alpha*d[self.iterate.Ik][i], xk[self.iterate.Ik][i] + alpha*(temp[i] - tempx[i]),'\n')
            '''
            self.xk = pi(xk - alpha * d)
            xk = self.xk
            self.iterates.append(self.iterate)
            self.iterate = PN_lcp_iterate(xk, A, b, bvals=self.bvals)
            self.num_iterations += 1
            residuals.append(np.linalg.norm(self.iterate.Fomega, np.inf))
            print('iteration:', str(self.num_iterations))
            print('residual:', residuals[-1], '\n')
            if 'show reduced space' in self.diagnostics:
                self.plot_active_set(xk, bounds, mx, my)
        self.iterate = PN_lcp_iterate(xk, A, b)
        self.iterates.append(self.iterate)
        print('convergence factor PN:', (residuals[-1] / residuals[0]) ** (1 / len(residuals)))
        # print('convergence factor l5:',(residuals[-1]/residuals[-5])**(1/5))
        # print('average size of red space:',np.mean(redspacesize))
        # print('red space % of whole:', np.mean(redspacesize)/len(xk))
        # print(redspacesize)
        return xk


class PN_lcp_iterate:
    '''
    Stores active set and inactive set, gradient and function value for each iterate.

    parameters:

        xk {vector} : current iterate; an (n x 0) or (n x 1) Numpy array.

        F {callable, tuple} : Objective function F. If F is affine, a tuple of the form (A, b) is accepted, and function
        values are computed as F(x) = Ax + b. Otherwise, F is a function that takes in an (n x 1) or (n x 0) Numpy array
        and returns an (n x 0) or (n x 1) Numpy array.

        gradF {callable, Nonetype} : Gradient gradF for F. If F is affine, gradF = A. Otherwise, gradF is a function which
        takes in an (n x 1) or (n x 0) Numpy array and returns an (n x n) Numpy array.

    attributes:

        Fk {vector} : F(xk), or the objective function value at xk. An (n x 1) or (n x 0) Numpy array.

        gradFk {maxtrix} : Jacobian matrix gradF(xk) for the function F at the current iterate. An (n x n) Numpy array.

        Ak {array} : Active set of indices for the current iterate.

        Ik {array} : Inactive set of indices for the current iterate.
    '''

    def __init__(self, xk, A, b, bvals=[]):

        self.Ik, self.Ak = [], []
        self.xk = xk
        self.b = b
        self.F = b - A.dot(xk)
        for i in range(len(xk)):
            if (xk[i] == 0 and self.F[i] > 0):
                self.Ak.append(i)
            else:
                self.Ik.append(i)
        self.search_dir = self.compute_Factive(A)
        self.Fomega = self.compute_Fomega()
        self.linear_solver_time = 0.0
        self.line_search_time = 0.0
        self.red_space_size = len(self.Ik)
        self.error = np.linalg.norm(self.Fomega, np.inf)
        self.alpha = 1
        self.line_search_fail = 0

    def compute_Fomega(self):

        x, F = self.xk, self.F
        Fomega = np.minimum(F, 0.0)
        Fomega[x > 0.0] = F[x > 0.0]
        return Fomega

    def compute_Factive(self, A):
        A_inv = spsolve(A, scipy.sparse.spdiags(-np.ones(len(self.xk)), [0], len(self.xk), len(self.xk)))
        A_inv[self.Ak, :] = 0.0
        A_inv[:, self.Ak] = 0.0
        d = A_inv.dot(self.F)
        for i in range(len(self.xk)):
            if (self.xk[i] == 0 and d[i] > 0):
                self.Ak.append(i)
                A_inv[i, :] = 0.0
                A_inv[:, i] = 0.0
        d = A_inv.dot(self.F)


        return d
