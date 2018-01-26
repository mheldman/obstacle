import numpy as np
from scipy.sparse.linalg import cg

'''
Reduced space algorithm for nonlinear complementarity problems (NCPs) of the following form:

        F(x) >= 0;
           x >= 0;
    x^T[F(x)] = 0,
       
where F: R^n to R^n.

For linear complementarity problems, f(x) = Ax + b is affine, and the problem reduces to:

     Ax - b >= 0;
          x >= 0;
 x^T(Ax - b) = 0.

Note that A is square.

The algorithm uses a modified Newton's method. On each iteration, an active set of indices is computed, defined by

    Active(x) = {i in {1,...,n} : x_i = 0 and F_i(x) > 0}

and an inactive set
    
    I(x) = {1,...,n} \ Active(x) = {i in {1,...,n} : x_i > 0 or F_i(x) <= 0}.

The active set encodes the indices where the nonnegativity constraints on the variables are active, so that 
the function value can be ignored. Once these sets are computed, the reduced Newton's method is applied; the gradient
grad(F)(xk) at the current iterate xk is computed, and a next search direction is found by taking a Newton step in the 
space of inactive constraints by approximately solving:

        [grad(F)(xk)]_{I(xk), I(xk)}d = -F_{I(xk)}(xk)

and stepping in the reduced space, i.e.

        xk_{I(xk), I(xk)} = xk_{I(xk), I(xk)} + d.
        
A line search the produces the next iterate.

Reference:

S.J. Benson, T.S. Munson, Flexible complementarity solvers for large-scale applications, 
Optim. Methods Softw. 21 (1) (2006) 155–168.

 '''


class rspmethod_solver:

    '''
    parameters:

    F {tuple,

    '''

    def __init__(self, F, gradF, tol, sigma, beta, gamma, maxiters, k, iterate, n):

        self.F = F
        self.gradF = gradF
        self.tol = tol
        self.sigma = sigma
        self.beta = beta
        self.gamma = gamma
        self.iterate = iterate
        self.dim = n





class rsp_iterate:

    '''
    Stores active set and inactive set, gradient and function value for each iterate.

    parameters:

    xk {vector} : current iterate
    F {callable, tuple} : Function F. If F is affine, a tuple of the form (A, b) is accepted, and function values
    are computed as F(x) = Ax + b.
    gradF {callable, Nonetype} : Gradient gradF for F. If F is affine, gradF = A.

    attributes:

    Fk {vector} : F(xk), where xk is the current iterate
    gradFk {maxtrix} : Jacobian matrix gradF(xk) at the current iterate
    Ak {array} : active set for the current iterate
    Ik {array} : inactive set for the current iterate

    '''

    def __init__(self, xk, F, gradF = None):

        if callable(F):
            self.Fk = F(xk)
            self.gradFk = gradF(xk)
        else:
            self.Fk = F[0].dot(xk) + F[1]
            self.gradFk = F[0]

        self.Ik = []
        self.Ak = []

        for i in range(0, len(xk)):
            if xk[i] = 0 and self.F[i] > 0:
                self.Ak.append(i)
            else:
                self.Ik.append(i)












def Fomega(A, x):
    n = len(x)
    y = np.zeros(n)
    y[[x > 0]] = A[[x > 0]]
    y[[x <= 0]] = np.minimum(A[[x <= 0]], 0.0)
    return y


def pi(x):
    y = x
    y[[x < 0]] = 0
    return y

def reducedspace(F, gradF, x0, tol = 10**-5, exact = True, sigma=10 ** -4, beta = .5, gamma = 10 ** -12):
    n = len(x0)
    k = 0
    xk = x0
    A = F(xk)
    FO = Fomega(A, xk)
    pik = pi(x0)
    while np.linalg.norm(FO, np.inf) > tol and k < 100:  # might use ||[x1*F1, x2*F2, ..., xn*Fn]||_inf
        k += 1
        Axk = []
        Ixk = []
        for i in range(0, n):
            if xk[i] == 0 and A[i] > 0:
                Axk.append(i)
            else:
                Ixk.append(i)
        d = np.zeros(n)
        temp = gradF(xk)
        m = len(Ixk)
        B = np.zeros((m, m))
        for i in range(0, m):
            for j in range(0, m):
                B[i, j] = temp[Ixk[i], Ixk[j]]
        dIxk, info = cg(B, -A[Ixk], tol=10 ** -5)
        j = 0
        for i in Ixk:
            d[i] = dIxk[j]
            j += 1

        alpha = beta
        fail = False
        Ak = F(pik)
        while np.linalg.norm(Fomega(Ak, pik)) > (1 - sigma * alpha) * np.linalg.norm(FO):
            pik = pi(xk + alpha * d)
            Ak = F(pik)
            alpha *= beta
            if alpha < gamma:
                fail = True
                break

        if fail:
            alpha = beta
            d = -F(xk)
            while np.linalg.norm(Fomega(Ak, pik)) > (1 - sigma * alpha) * np.linalg.norm(FO):
                alpha = alpha * beta
                pik = pi(xk + alpha * d)
                Ak = F(pik)
            if beta < gamma:
                print('Could not provide sufficient decrease. Process terminated iteration', k)
                return xk

        xk = pik
        A = F(xk)
        FO = Fomega(A, xk)
    print('\n', 'F(xk)*xk =', np.transpose(F(xk)).dot(xk))
    return xk

