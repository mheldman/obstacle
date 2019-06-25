import numpy as np
from scipy.sparse.linalg import spsolve, isolve
import pyamg
from time import time

__all__ = ['rspmethod_lcp_solver', 'multilevel_rsp_solver', 'level', 'rsp_lcp_iterate']

'''
Reduced space algorithm for linear complementarity problems (LCPs).

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

    Active(x) = {i in {1,...,n} : x_i = 0 and [A(x)]_i > 0}

and an inactive set
    
    I(x) = {1,...,n} \ Active(x) = {i in {1,...,n} : x_i > 0 or [A(x)]_i <= 0}.

The active set encodes the indices where the nonnegativity constraints on the variables are active, so that 
the function value can be ignored. Once these sets are computed, the reduced Newton's method is applied; the gradient
grad(Ax - b) = A at the current iterate xk is computed, and a next search direction is found by taking a Newton step in 
the space of inactive constraints by approximately solving:

        [Axk]_{I(xk), I(xk)}d = -(Axk - b)_{I(xk)}(xk)

and stepping in the reduced space, i.e.

        xk_{I(xk), I(xk)} = xk_{I(xk), I(xk)} + alpha*d.        

Reference:

S.J. Benson, T.S. Munson, Flexible complementarity solvers for large-scale applications, 
Optim. Methods Softw. 21 (1) (2006) 155_168.
 '''


def pi(x):
    y = np.zeros_like(x)
    y[x > 0] = x[x > 0]
    return y

class rspmethod_lcp_solver:

    '''
    parameters:

    F {tuple, callable}

    '''

    def __init__(self, A, b, tol, maxiters, bvals, diagnostics):

        self.A = A
        self.b = b
        self.tol = tol
        self.maxiters = maxiters
        self.num_iterations = 0
        self.xk = np.zeros_like(b)
        self.iterate = rsp_lcp_iterate(self.xk, A, b)
        self.fixed_vals = bvals
        self.iterates = []
        self.bvals = bvals
        self.diagnostics=diagnostics

    def compute_Fomega(self, x = None):

        if x is None:

            x = self.xk
            Fomega = np.minimum(F, 0.0)
            Fomega[x > 0] = F[x > 0]
            Fomega[self.fixed_vals] = 0.0
            self.iterate.Fomega = Fomega
            return Fomega

        else:

            F = self.b - self.A.dot(x)
            Fomega = np.minimum(F, 0.0)
            Fomega[x > 0] = F[x > 0]
            Fomega[self.fixed_vals] = 0.0
            return Fomega


    def solve(self, mx, my=None, bounds=(-1.,1,-1.,1.), init_iterate = None, line_search_params = None, linear_solver = 'cg', preconditioner = 'amg'):

        A = self.A
        b = self.b

        if line_search_params is None:
            beta = .5
            sigma = 10e-4
            gamma = 10e-12
        
        else:
            beta, sigma, gamma = line_search_params

        if init_iterate is not None:
            self.iterate = rsp_lcp_iterate(pi(init_iterate), A, b, fixed_vals=self.fixed_vals)
            self.xk = self.iterate.xk
            xk = self.xk
        
        else:
            self.iterate = rsp_lcp_iterate(pi(spsolve(A, b)), A, b, fixed_vals = self.fixed_vals)
            self.xk = self.iterate.xk
            xk = self.xk
        self.iterates.append(self.iterate)
        
        if self.num_iterations % 1 == 0:
            print('Iteration ' + str(self.num_iterations) + ' | ' + '||Fomega||_inf = ' + str(np.linalg.norm(self.iterate.Fomega, np.inf)))
        origerr = self.iterate.error
        
        while np.linalg.norm(self.iterate.Fomega, np.inf)/origerr > self.tol and self.num_iterations < self.maxiters:

            #self.iterate.xk = pgs(A, xk, b)
            xk = self.iterate.xk

            F_active, gradF_active = self.iterate.compute_Factive(A)
            tstart = time()
            Ik = self.iterate.Ik
            
            if preconditioner == 'splu':
                d = spsolve(gradF_active, -F_active)
            
            elif linear_solver == 'amg':
                currenterr = self.iterate.error
                mls = pyamg.smoothed_aggregation_solver(gradF_active, B=None)
                residuals=[]
                tol=1e-10*origerr/currenterr
                d = mls.solve(-F_active, tol=tol, residuals=residuals)
                print('\nksp convergence history:')
                for i in range(len(residuals)):
                  print('residual ' + str(i + 1) + ': ' + str(residuals[i]))
                print('\n')

            
            elif linear_solver == 'cg':
              from scipy.sparse.linalg import cg
              
              if preconditioner == 'ilu':
                from scipy.sparse.linalg import LinearOperator, spilu
                ilu = spilu(gradF_active.tocsc())
                Mx = lambda x: ilu.solve(x)
                shape = gradF_active.shape
                M = LinearOperator(shape, Mx)
                d = cg(gradF_active, -F_active, x0=xk[Ik], M=M)[0]
            
              elif preconditioner == 'amg':
              
                currenterr = self.iterate.error
                mls = pyamg.smoothed_aggregation_solver(gradF_active, B=None)
                residuals=[]
                tol=1e-10*origerr/currenterr
                d = mls.solve(-F_active, tol=tol, accel='cg', residuals=residuals)
                print('\nksp convergence history:')
                for i in range(len(residuals)):
                  print('residual ' + str(i + 1) + ': ' + str(residuals[i]))
                print('\n')
            
              elif preconditioner == 'none':
                d = cg(gradF_active, -F_active, x0=xk[Ik])[0]
                
              '''
            elif preconditioner == 'gmg':
            
                currenterr = self.iterate.error
                mls = pyamg.smoothed_aggregation_solver(gradF_active, B=None)
                residuals=[]
                tol=1e-10*origerr/currenterr
                d = mls.solve(-F_active, tol=tol, accel='cg', residuals=residuals)
                print('\nksp convergence history:')
                for i in range(len(residuals)):
                  print('residual ' + str(i + 1) + ': ' + str(residuals[i]))
                print('\n')
            '''
              
              

            self.iterate.linear_solver_time = time() - tstart
            self.iterate.search_dir = np.zeros_like(xk)
            self.iterate.search_dir[self.iterate.Ik] = d
            d = self.iterate.search_dir
            alpha, fail = 1, 0
            tstart = time()

            
            while np.linalg.norm(self.compute_Fomega(pi(xk + alpha*d)))\
                                                > (1 - sigma * alpha) * np.linalg.norm(self.iterate.Fomega):
                alpha *= beta
                if alpha < gamma:
                    fail += 1
                    self.iterate.search_dir = -self.iterate.F
                    d = self.iterate.search_dir
                    alpha = 1

                if fail == 2:
                    break

            self.iterate.line_search_time = time() - tstart
            self.iterate.alpha = alpha
            self.iterate.line_search_fail = fail
            xk = pi(xk + alpha*d)
            self.xk = xk

            self.iterates.append(self.iterate)
            self.iterate = rsp_lcp_iterate(self.xk, A, b, fixed_vals=self.fixed_vals)
            self.num_iterations += 1
            if self.num_iterations % 1 == 0:
              print('Iteration ' + str(self.num_iterations) + ' | ' + '||Fomega||_inf = ' + str(np.linalg.norm(self.iterate.Fomega, np.inf)))
        return xk

class multilevel_rsp_solver:
  
  def __repr__(self):
    output = 'Multilevel reduced space solver\n'
    output += 'Number of levels = ' + str(len(self.levels)) + '\n'
    output += 'Fine grid size (' + str((self.levels[0].mx + 2)) + ' x ' + str((self.levels[0].my + 2)) + ')\n'
    output += str(self.levels[0].mx * self.levels[0].my) + ' fine grid unknowns\n'
    output += 'Coarse grid size (' + str(self.coarse_mx + 2) + ' x ' + str(self.coarse_my + 2) + ')\n'
    output += str(self.coarse_mx * self.coarse_my) + ' coarse grid unknown(s)\n'
    return output

  def __init__(self, levels, coarse_mx, coarse_my, linear_solver='spsolve', preconditioner='amg', diagnostics=()):
    
      self.levels = levels
      self.linear_solver = linear_solver
      self.preconditioner = preconditioner
      self.coarse_mx = coarse_mx
      self.coarse_my = coarse_my
      self.residuals = []
      self.diagnostics = diagnostics

  def solve(self, b, tol, maxiters):
    u0 = None
    i = 0
    for lvl in self.levels:
        if i == len(self.levels) - 1:
          tstart = time()
        mx, my = lvl.mx, lvl.my
        print('level ' + str(i + 1) + ' |' + ' grid size (' + str(mx + 2) + ' x ' + str(my + 2) + ')')
        A, b = lvl.A, lvl.b
        if i != 0:
          u0[lvl.bndry_pts] = b[lvl.bndry_pts]
        lvl_solver = rspmethod_lcp_solver(A, b, tol, maxiters, lvl.bndry_pts, self.diagnostics)
        u = lvl_solver.solve(lvl.mx, my=lvl.my, bounds=lvl.bounds, init_iterate = u0, line_search_params = None, linear_solver = self.linear_solver, preconditioner = self.preconditioner)
        i += 1
        if i < len(self.levels):
          u0 = pi(lvl.P.dot(u))
        else:
          print('fine grid solution time: ' + str(time() - tstart))
          return u
        print('\n')

class level:
  '''
    Stores one level of the multigrid hierarchy and implements recursive geometric multigrid solvers. The current
    implementation assumes the PDE is being solved on a rectangular region with a discretization in horizontal rows
    and vertical columns. This class functions as a struct.
    
    Attributes:
    
    mx {int}: Number of unknowns in the horizontal direction on the current level
    my {int}: Number of unknowns in the vectical direction on the current level
    x1, x2, y1, y2 {int}: Determines the region where the problem is being solved. x1 and x2 are the bounds
    in the horizontal direction, y1 and y2 are the bounds in the vertical direction.
    hx {float}: Grid spacing the horizontal direction on the current level. Given by (x2 - x1) / (mx + 1)
    hy {float}: Grid spacing in the vertical direction on the current level. Given by (y2 - y1) / (my + 1)
    A {spmatrix}: A square spmatrix of size (mx + 2)*(my + 2). The differential operator on the current
    level.
    R {spmatrix}: Restriction matrix. Transfers the problem to the next coarser grid.
    P {spmatrix}: Prolongation matrix. Transfers the problem to the next finer grid.
    
    '''
      
  def __init__(self, mx, my, A, b, P, x1, x2, y1, y2, bndry_pts):
    
    if callable(A):
      self.A = A(mx, my, x1, x2, y1, y2)
    else:
      self.A = A
    if callable(P):
      self.P = P(mx, my)
    else:
      self.P = P
    self.bounds = (x1, x2, y1, y2)
    self.mx = mx
    self.my = my
    self.hx = (x2 - x1) / (mx + 1)
    self.hy = (y2 - y1) / (my + 1)
    self.bndry_pts = bndry_pts
    self.b = b



class rsp_lcp_iterate:


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

    def __init__(self, xk, A, b, fixed_vals = []):

        self.Ik, self.Ak = [], []
        self.F = b - A.dot(xk)
        for i in range(len(xk)):
            if (xk[i] == 0 and self.F[i] > 0) or i in fixed_vals:
                self.Ak.append(i)
            else:
                self.Ik.append(i)

        self.search_dir = np.zeros_like(xk)
        self.xk = xk
        self.F_active, self.gradF_active = self.compute_Factive(A)
        self.Fomega = self.compute_Fomega(fixed_vals = fixed_vals)
        self.linear_solver_time = 0.0
        self.line_search_time = 0.0
        self.red_space_size = len(self.Ik)
        self.error = np.linalg.norm(self.Fomega, np.inf)
        self.alpha = 1
        self.line_search_fail = 0


    def compute_Fomega(self, fixed_vals = []):

        xk, F = self.xk, self.F
        Fomega = np.minimum(F, 0.0)
        Fomega[xk > 0.0] = F[xk > 0.0]
        Fomega[fixed_vals] = 0.0
        return Fomega

    def compute_Factive(self, A):

        gradF_active = -A[self.Ik, :][:, self.Ik]
        F_active = self.F[self.Ik]
        return F_active, gradF_active.tocsr()


'''
 self.iterate.xk = pgs(A, xk, b)
            xk = self.iterate.xk

            #F_active, gradF_active = self.iterate.compute_Factive(A)
            tstart = time()

            if linear_solver == 'splu':
                red_xk = pi(spsolve(A[self.iterate.Ik, :][:, self.iterate.Ik], -b[self.iterate.Ik]))
                xk[self.iterate.Ik] = red_xk
                self.xk = xk
                self.iterate.xk = xk
            elif linear_solver == 'amg':
                d = pyamg.solve(gradF_active, -F_active, verb=True, tol=1e-12)
            self.iterate.linear_solver_time = time() - tstart


            self.iterate.search_dir = np.zeros_like(xk)
            self.iterate.search_dir[self.iterate.Ik] = d
            d = self.iterate.search_dir
            alpha, fail = 1, 0
            tstart = time()

            while np.linalg.norm(self.compute_Fomega(pi(xk + alpha*d)))\
                                                > (1 - sigma * alpha) * np.linalg.norm(self.iterate.Fomega):
                alpha *= beta
                if alpha < gamma:
                    fail += 1
                    self.iterate.search_dir = -self.iterate.F
                    d = self.iterate.search_dir
                    alpha = 1

                if fail == 2:
                    break

            self.iterate.line_search_time = time() - tstart
            self.iterate.alpha = alpha
            self.iterate.line_search_fail = fail
            self.xk = pi(xk + alpha*d
            xk = self.xk

            self.iterates.append(self.iterate)
            self.iterate = rsp_lcp_iterate(xk, A, b, fixed_vals=self.fixed_vals)
            self.num_iterations += 1
            if self.num_iterations % 1 == 0:
                print('Iteration:', str(self.num_iterations))
                print('||Fomega||_inf =', np.linalg.norm(self.iterate.Fomega, np.inf), '\n')

        return xk


'''


