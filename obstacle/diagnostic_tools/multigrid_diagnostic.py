# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse.linalg import spsolve
import scipy.sparse
from GS_diagnostic import pgs, gs
import matplotlib.pyplot as plt
from time import time
from mpl_toolkits.mplot3d import Axes3D
from ReducedSpace_diagnostic import rspmethod_lcp_solver

class multigrid_solver:

    '''
    Stores the multigrid hierarchy and implements recursive geometric multigrid solvers. The current implementation
    assumes the PDE is being solved on a rectangular region with a discretization in horizontal rows and vertical
    columns.

    Attributes:

        levels {iterable} : Contains level objects, which store the information for each level
        coarse_solver {callable} : Exact solver for the coarse grid.
        Inputs to coarse_solver should have the form (A, x, b, **kwargs)
        coarse_mx {int} : Number of unknowns in the horizontal direction
        coarse_my {int} : Number of unknowns in the vectrical direction
        level {level} : Stores the current level during the multigrid iteration. Initially set to the
        finest grid.
        smoother {callable} : Smoother to be used on each grid. Inputs should have the form (A, x, b, **kwargs),
        where A is a sparse square matrix, x is the current iterate, to be used as the intial guess for the smoother,
        and b is the right-hand side vector. The system is square with size (self.level.mx + 2)*(self.level.my + 2).

    Methods:

        lvl_solve: Recursively solves the discretized PDE.
        solve: Initializes the system and calls lvl_solve to solve the PDE


    '''

    def __repr__(self):
        output = 'Multigrid solver\n'
        output += 'Number of levels = ' + str(len(self.levels)) + '\n'
        output += 'Fine grid size (' + str((self.levels[0].mx + 2)) + ' x ' + str((self.levels[0].my + 2)) + ')\n'
        output += str(self.levels[0].mx * self.levels[0].my) + ' fine grid unknowns\n'
        output += 'Coarse grid size (' + str(self.coarse_mx + 2) +  ' x ' + str(self.coarse_my + 2) + ')\n'
        output += str(self.coarse_mx * self.coarse_my) + ' coarse grid unknown(s)\n'
        return output

    def __init__(self, levels, coarse_mx, coarse_my, smoother, coarse_solver=spsolve, diagnostics=()):

        self.levels = levels
        self.coarse_solver = coarse_solver
        self.coarse_mx = coarse_mx
        self.coarse_my = coarse_my
        self.level = self.levels[0]
        self.smoother = smoother
        self.residuals = []
        self.diagnostics=diagnostics
        self.smoothing_times = []
        self.transfer_times=[]

    def lvl_solve(self, lvl, u, b, cycle, smoothing_iters=1):

        self.level = self.levels[lvl]
        A = self.level.A
        R = self.level.R
        if 'show levels' in self.diagnostics:
            print(lvl * "    " + "Grid " + str(lvl) + ", mx = " + str(self.level.mx) + ", my = " + str(self.level.my))

        for i in range(smoothing_iters):
            tstart = time()
            self.smoother(A, u, b, maxiters=1)
            self.smoothing_times.append(tstart-time())
        tstart=time()
        r = b - A.dot(u)
        coarse_b = R.dot(r)
        self.transfer_times.append(time() - tstart)

        if lvl < len(self.levels) - 2:
            coarse_u = np.zeros_like(coarse_b)
            if cycle == 'W':
                self.lvl_solve(lvl + 1, coarse_u, coarse_b, cycle)
                self.lvl_solve(lvl + 1, coarse_u, coarse_b, cycle)

            elif cycle == 'V':
                self.lvl_solve(lvl + 1, coarse_u, coarse_b, cycle)

            elif cycle == 'FV':
                self.lvl_solve(lvl + 1, coarse_u, coarse_b, cycle)
                self.lvl_solve(lvl + 1, coarse_u, coarse_b, 'V')

            elif cycle == 'FW':
                self.lvl_solve(lvl + 1, coarse_u, coarse_b, cycle)
                self.lvl_solve(lvl + 1, coarse_u, coarse_b, 'W')

            elif cycle == 'fmgV':
                coarse_b2 = R.dot(b)
                self.lvl_solve(lvl + 1, coarse_u, coarse_b2, cycle)
                self.lvl_solve(lvl + 1, coarse_u, coarse_b, 'V')

        else:
            if 'show levels' in self.diagnostics:
                print((lvl + 1) * "    " + "Grid " + str(lvl + 1) + ", mx = " + str(self.coarse_mx) + ", my = " + str(
                self.coarse_my))
            coarse_u = self.coarse_solver(self.levels[-1].A, coarse_b)

        P = self.levels[lvl + 1].P
        tstart=time()
        u += P.dot(coarse_u)
        self.transfer_times.append(time() - tstart)
        #c = A.diagonal()
        #u[c == 1.0] = b[c == 1.0]
        for i in range(smoothing_iters):
            tstart = time()
            self.smoother(A, u, b)
            self.smoothing_times.append(time() - tstart)
        self.level = self.levels[lvl]
        if 'show levels' in self.diagnostics:
            print(lvl * "    " + "Grid " + str(lvl) + ", mx = " + str(self.level.mx) + ", my = " + str(
            self.level.my))

    def solve(self, b, u0=None, cycle='FV', tol=1e-8, maxiters=50, smoothing_iters=1):

        if u0 is None:
            u0 = np.zeros_like(b)

        u = np.array(u0)
        self.residuals.append(np.linalg.norm(b - self.level.A.dot(u), np.inf))
        z = 0
        while self.residuals[-1] > tol and z < maxiters:
            print('gmg residual iteration ' + str(z) + ': ' + str(self.residuals[-1]))
            self.lvl_solve(0, u, b, cycle, smoothing_iters=smoothing_iters)
            self.residuals.append(np.linalg.norm(b - self.level.A.dot(u), np.inf))
            z += 1
        residuals=self.residuals
        print('gmg final residual iteration ' + str(z) + ': ' + str(residuals[-1]))
        print('convergence factor gmg: ' + str((residuals[-1]/residuals[0])**(1.0/len(residuals))) + '\n')
        return u


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

    def __init__(self, mx, my, A, R, P, x1, x2, y1, y2, bndry_pts):

        if callable(A):
            self.A = A(mx, my, x1, x2, y1, y2)
        else:
            self.A = A

        if callable(R):
            self.R = R(mx, my)
        else:
            self.R = R

        if callable(P):
            self.P = P(mx, my)
        else:
            self.P = P
        self.bounds = (x1,x2,y1,y2)
        self.mx = mx
        self.my = my
        self.hx = (x2 - x1) / (mx + 1)
        self.hy = (y2 - y1) / (my + 1)
        self.bndry_pts = bndry_pts

def compute_Fomega(x, F): #merit function (residual) for LCP
        Fomega = np.minimum(F, 0.0)
        bool_array=(x > 1e-16)
        Fomega[bool_array] = F[bool_array]
        return Fomega

class linear_pfas_solver:

    '''
    Stores the multigrid hierarchy and implements the projected full-approximation scheme (developed in [1]) for the
    solution of linear complementarity problems arising from free boundary problems. The free-boundary problem should
    occur on a rectangular region with a discretization in horizontal rows and vertical columns.

    Attributes:

        levels {iterable} : Contains level objects, which store the information for each level
        coarse_solver {callable} : Exact solver for the coarse grid.
        Inputs to coarse_solver should have the form (A, x, b, **kwargs)
        coarse_mx {int} : Number of unknowns in the horizontal direction
        coarse_my {int} : Number of unknowns in the vectrical direction
        level {level} : Stores the current level during the multigrid iteration. Initially set to the
        finest grid.
        smoother {callable} : Smoother to be used on each grid. Inputs should have the form (A, x, b, **kwargs),
        where A is a sparse square matrix, x is the current iterate, to be used as the initial guess for the smoother,
        and b is the right-hand side vector. The system is square with size (self.level.mx + 2)*(self.level.my + 2).

    Methods:

        lvl_solve: Recursively solves the discretized free boundary problem
        solve: Initializes the system and calls lvl_solve to solve the free boundary problem

    Sources:

    [1] Achi Brandt and Colin W. Cryer. Multigrid algorithms for the solution of linear complementarity problems
        arising from free boundary problems. Siam Journal on Scientific and Statistical Computing, 4(4):655–684, 1983.

    '''

    def __repr__(self):
        output = 'PFAS solver\n'
        output += 'Number of levels = ' + str(len(self.levels)) + '\n'
        output += 'Fine grid size (' + str((self.levels[0].mx + 2)) + ' x ' + str((self.levels[0].my + 2)) + ')\n'
        output += str(self.levels[0].mx * self.levels[0].my) + ' fine grid unknowns\n'
        output += 'Coarse grid size (' + str((self.coarse_mx + 2)) + ' x ' + str((self.coarse_my + 2)) + ')\n'
        output += str(self.coarse_mx * self.coarse_my) + ' coarse grid unknown(s)\n'
        return output


    def __init__(self, levels, coarse_mx, coarse_my, smoother, coarse_solver=pgs, diagnostics=None):

        self.levels = levels
        self.coarse_solver = coarse_solver
        self.coarse_mx = coarse_mx
        self.coarse_my = coarse_my
        self.level = self.levels[0]
        self.smoother = smoother
        self.mu = .15
        self.diagnostics=diagnostics
        self.residuals=[]
        self.bndry_pts = []
        self.smoothing_times=[]
        self.transfer_times=[]

    def lvl_solve(self, lvl, u, b, cycle, smoothing_iters=1):

        self.level = self.levels[lvl]
        if 'show levels' in self.diagnostics:
            print(lvl * "    " + "Grid " + str(lvl) + ", mx = " + str(self.level.mx) + ", my = " + str(self.level.my))
        A = self.level.A
        for i in range(smoothing_iters):
            tstart = time()
            self.smoother(A, u, b)
            self.smoothing_times.append(time()-tstart)
        '''
        Fomega = abs(compute_Fomega(u, b - A.dot(u)))
        args = np.argsort(-Fomega)
        for j in range(3):
            for k in range(len(args)):
                i = args[k]
                u[i] = max(0.0, (1 / A[i, i]) * (b[i] - A[i, :].dot(u) + A[i, i] * u[i]))
                if i > .1*len(u):
                    break
        mx = self.level.mx
        my = self.level.my
        self.bndry_pts = []
        kk = lambda i, j: j * (mx + 2) + i
        for j in range(0, my + 2):
            for i in range(0, mx + 2):
                k = kk(i, j)
                if j in [1, 2, my, my - 1] or i in [1, 2, mx, mx - 1]:
                    self.bndry_pts.append(k)

        for j in range(10):
            for i in self.bndry_pts:
                u[i] = max(0.0, (1/A[i, i])*(b[i] - A[i, :].dot(u) + A[i, i] * u[i]))
        '''
        R = self.level.R
        tstart = time()
        coarse_u = R.dot(u)
        coarse_b = R.dot(b - A.dot(u))
        coarse_A = self.levels[lvl + 1].A #needs at least two levels
        coarse_b = coarse_b + coarse_A.dot(coarse_u)
        self.transfer_times.append(time()-tstart)
        if lvl < len(self.levels) - 2:

            if cycle == 'W':
                self.lvl_solve(lvl + 1, coarse_u, coarse_b, cycle)
                self.lvl_solve(lvl + 1, coarse_u, coarse_b, cycle)

            elif cycle == 'V':
                self.lvl_solve(lvl + 1, coarse_u, coarse_b, cycle)

            elif cycle == 'FV':
                self.lvl_solve(lvl + 1, coarse_u, coarse_b, 'FV')
                self.lvl_solve(lvl + 1, coarse_u, coarse_b, 'V')

            elif cycle == 'FW':
                self.lvl_solve(lvl + 1, coarse_u, coarse_b, 'FW')
                self.lvl_solve(lvl + 1, coarse_u, coarse_b, 'W')

            elif cycle == 'fmgV':
                coarse_b2 = R.dot(b)
                self.lvl_solve(lvl + 1, coarse_u, coarse_b2, 'fmgV')
                self.lvl_solve(lvl + 1, coarse_u, coarse_b, 'V')

        else:
            if 'show levels' in self.diagnostics:
                print((lvl + 1) * "    " + "Grid " + str(lvl + 1) + ", mx = " + str(self.coarse_mx) + ", my = " + str(
                    self.coarse_my))
            du = 1.0
            tstart=time()
            while du > 10 ** -12:
                uold = coarse_u.copy()
                self.smoother(coarse_A, coarse_u, coarse_b)
                du = (self.coarse_mx + 1) * np.linalg.norm(uold - coarse_u)
            self.smoothing_times.append(time()-tstart)

        P = self.levels[lvl + 1].P
        tstart = time()
        u += P.dot(coarse_u - R.dot(u))
        self.transfer_times.append(time() - tstart)
        self.level = self.levels[lvl]

        if 'show levels' in self.diagnostics:
            print(lvl * "    " + "Grid " + str(lvl) + ", mx = " + str(self.level.mx) + ", my = " + str(self.level.my))

        for i in range(smoothing_iters):
            tstart=time()
            self.smoother(A, u, b)
            self.smoothing_times.append(time() - tstart)
        '''
        mx = self.level.mx
        my = self.level.my
        self.bndry_pts = []
        kk = lambda i, j: j * (mx + 2) + i
        for j in range(0, my + 2):
            for i in range(0, mx + 2):
                k = kk(i, j)
                if j == 1 or j == my or i == 1 or i == mx:
                    self.bndry_pts.append(k)
        for j in range(10):
            for i in self.bndry_pts:
                u[i] = max(0.0, (1 / A[i, i]) * (b[i] - A[i, :].dot(u) + A[i, i] * u[i]))

        for j in range(3):
            for i in self.bndry_pts:
                u[i] = max(0.0, (1/A[i, i])*(b[i] - A[i, :].dot(u) + A[i, i] * u[i]))
        
        args = np.argsort(-abs(compute_Fomega(u, b - A.dot(u))))
        for j in range(3):
            for k in range(len(args)):
                i = args[k]
                u[i] = max(0.0, (1 / A[i, i]) * (b[i] - A[i, :].dot(u) + A[i, i] * u[i]))
                if i > .1 * len(u):
                    break
        '''




    def plot_active_set(self, u):

        x1, x2, y1, y2 = self.level.bounds
        kk = lambda i, j: (self.level.mx + 2) * i + j
        Z = np.zeros((self.level.mx + 2, self.level.my + 2))
        for i in range(0, self.level.my + 2):
            for j in range(0, self.level.mx + 2):
                k = kk(i, j)
                Z[j, i] = u[k]
        X = np.linspace(x1, x2, self.level.mx + 2)
        Y = np.linspace(y1, y2, self.level.my + 2)
        A, B = np.meshgrid(X, Y)
        A, B = np.transpose(A), np.transpose(B)
        plt.plot(A[[Z < 1e-10]], B[[Z < 1e-10]], 'o',color='k')
        plt.ion()
        plt.xlim(x1, x2)
        plt.ylim(y1, y2)
        plt.show()
        plt.pause(2)
        plt.close('all')

    def active_set(self, u, r):
        return np.arange(len(r))[(u < 1e-16) & (r > 0.0)]



    def solve(self, b, u0=None, cycle='FV', tol=1e-8, maxiters=400, smoothing_iters=1):

        print('pfas solver maxiters: ' + str(maxiters))
        print('pfas solver residual tolerance: ' + str(tol) + '\n')
        if u0 is None:
            #u0 = pgs(self.level.A, np.zeros_like(b), b)
            #u0 = spsolve(self.level.A, b)
            #linear_solver = multigrid_solver(self.levels, self.coarse_mx, self.coarse_my, gs, coarse_solver=spsolve)
            print('computing pfas initial guess...' + '\n')
            #print(linear_solver)
            #u0 = linear_solver.solve(b, tol=tol*np.sqrt(len(b)))
            #u0 = spsolve(self.level.A, b)
            u0 = np.zeros_like(b)
            u0[self.levels[0].bndry_pts] = b[self.levels[0].bndry_pts]
        u = np.array(u0)
        u[u < 0.0] = 0.0
        residuals = []
        #tstart=time()
        r = b - self.level.A.dot(u)
        residuals.append(np.linalg.norm(compute_Fomega(u, r), np.inf))
        #self.transfer_times.append(time()-tstart)
        z = 0

        if 'show residuals' in self.diagnostics:
            print('pfas residual iteration ' + str(z) + ': ' + str(residuals[-1]))
            #print('(active set new) sym diff (active set old): ' + '-----')

        if 'show reduced space' in self.diagnostics:
            self.plot_active_set(u)
        active_set_old = set(self.active_set(u, r))
        active_set_new=active_set_old
        while residuals[-1]/residuals[0] > tol and z < maxiters:
            self.lvl_solve(0, u, b, cycle, smoothing_iters=smoothing_iters)
            z += 1
            tstart = time()
            r = b - self.level.A.dot(u)
            residuals.append(np.linalg.norm(compute_Fomega(u, r), np.inf))
            self.transfer_times.append(time() - tstart)

            if 'show residuals' in self.diagnostics:
                print('pfas residual iteration ' + str(z) + ': ' + str(residuals[-1]))

            if 'show reduced space' in self.diagnostics:
                self.plot_active_set(u)
            tstart=time()
            active_set_old = active_set_new
            active_set_new = set(self.active_set(u, r))
            self.transfer_times.append(time() - tstart)
            #print(len(active_set_new),  len(active_set_new.symmetric_difference(active_set_old)))
            #if z!=0 and residuals[-1] < 1.0:
                #sym_diff = len(active_set_new.symmetric_difference(active_set_old))
                #print('(active set new) sym diff (active set old): ' + str(sym_diff))
            #else:
             #   sym_diff = 1
             #   print('(active set new) sym diff (active set old): ' + '----------')

            if z != 0 and (active_set_new == active_set_old or (residuals[-1]/residuals[0])**(1.0/(len(residuals) - 1.0)) > .75): #z!=0 and residuals[-1]/residuals[-2] > .5:

                if (residuals[-1]/residuals[0])**(1.0/(len(residuals) - 1.0)) > .75:
                    print('solver diverged.. convergence factors > .75')
                else:
                    print('\n' + 'active set converged. solving BVP...')
                print('geometric convergence factor PFAS: ' + str((residuals[-1] / residuals[0]) ** (1.0 / (len(residuals) - 1.0))) + '\n')
                break

        gmg_called = False
        while residuals[-1]/residuals[0] > tol:
            if gmg_called:
                active_set_new = self.active_set(u, b1)
            if not gmg_called:
                b1 = b.copy()
                Alist = []
                for i in range(len(self.levels)):
                    Alist.append(self.levels[i].A.copy())
            gmg_called = True
            active_set_vec = np.zeros((self.level.mx + 2) * (self.level.my + 2))
            active_set_new = list(active_set_new)
            b[active_set_new] = 0.0
            active_set_vec[active_set_new] = 1
            # rsp_solver = rspmethod_lcp_solver(self.level.A, b, 1e-4, 25, diagnostics=self.diagnostics, bvals=self.level.bndry_pts)
            # u = rsp_solver.solve(init_iterate = u, linear_solver='amg', bounds=self.level.bounds, mx =self.level.mx, my=self.level.my)
            for i in range(len(self.levels)):
                self.levels[i].A = Alist[i]
                I_elim = -scipy.sparse.diags(active_set_vec - 1, 0, shape=self.levels[i].A.shape, format='csr')
                I_add = scipy.sparse.diags(active_set_vec, 0, shape=self.levels[i].A.shape, format='csr')
                self.levels[i].A = I_elim.dot(self.levels[i].A.dot(I_elim)) + I_add
                active_set_vec = self.levels[i].R.dot(active_set_vec)
            gmg_solver = multigrid_solver(self.levels, self.coarse_mx, self.coarse_my, gs, coarse_solver=spsolve)
            print(gmg_solver)
            u = gmg_solver.solve(b, u0=u, tol=residuals[0] * tol, cycle=cycle, smoothing_iters=smoothing_iters)
            u[u < 0.0] = 0.0
            b = b1.copy()
            residuals.append(np.linalg.norm(compute_Fomega(u, b - Alist[0].dot(u)), np.inf))
            for i in range(len(self.levels)):
                self.levels[i].A = Alist[i]
            print('pfas residual iteration ' + str(z) + ': ' + str(residuals[-1]))
            z += 1
            #for i in range(len(self.levels)):
            #    self.levels[i].A = Alist[i].copy()
            #active_set_new = self.active_set(u, b1)
            #active_set_vec = np.zeros((self.level.mx + 2) * (self.level.my + 2))
            #active_set_new = list(active_set_new)
            # print('total convergence factor: ', (residuals[-1]/residuals[0])**(1.0/(len(gmg_resid) + len(residuals) - 1.0)) )
            #b = b1.copy()
            #b[active_set_new] = 0.0
            self.smoothing_times += gmg_solver.smoothing_times
            self.transfer_times += gmg_solver.transfer_times
        #ite active_set_vec[active_set_new] = 1                                  rates = rsp_solver.iterates
        #residuals_rsp = []
        #for iterate in iterates:
         #   residuals_rsp.append(iterate.error)
        num_pfas = len(residuals) - 1
        if gmg_called:
            for i in range(len(gmg_solver.residuals)):
                residuals = residuals[0:num_pfas] + gmg_solver.residuals
        else:
            num_pfas = len(residuals)

        if residuals[-1]/residuals[0] < tol:
            print('\n' + 'convergence summary')
            print('-------------------')
            #residuals = residuals + residuals_rsp[2:len(residuals_rsp)]
            for i in range(len(residuals)):
                if i == 0:
                    print('pfas residual ' + str(i) + ': ' + str(residuals[i]) + ' ' * (22 + len(str(maxiters)) - len(str(residuals[i])) - len(str(i))) + 'convergence factor 0: ---------------')
                elif i < num_pfas:
                    print('pfas residual ' + str(i) + ': ' + str(residuals[i]) + ' ' * (22 + len(str(maxiters)) - len(str(residuals[i])) - len(str(i))) \
                          + 'convergence factor ' + str(i) + ': ' + str(residuals[i] / residuals[i - 1]))
                elif i > num_pfas:
                    print('gmg residual  ' + str(i - 1) + ': ' + str(residuals[i]) + ' '*(22 + len(str(maxiters)) - len(str(residuals[i])) - (len(str(i - 1))))\
                                                    + 'convergence factor '+str(i - 1) + ': ' + str(residuals[i]/residuals[i-1]))

        self.residuals = residuals



        #plt.ioff()
        #plt.semilogy(np.arange(len(residuals)), residuals, '*-')
        #plt.xlabel('Iteration')
        #plt.ylabel('Residual')
        #plt.show()
        '''
        if 'show reduced space' in self.diagnostics:
            self.plot_active_set(u)

        if 'show residuals' in self.diagnostics:
            print('final residual:', residuals[-1])

        if 'geometric convergence factor' in self.diagnostics:
            print('geometric convergence factor: ', (residuals[-1]/residuals[0])**(1.0/len(residuals)))
           #print('geometric convergence factor f3: ', (residuals[2] / residuals[0]) ** (1.0 / 3))
        '''

        if 'show reduced space' in self.diagnostics:
            mx, my = self.level.mx, self.level.my
            x1, x2, y1, y2 = self.level.bounds
            kk = lambda i,j: (self.level.mx + 2) * i + j
            Z, Z1 = np.zeros((mx + 2, my + 2)), np.zeros((mx + 2, my + 2))
            for i in range(0, my + 2):
                for j in range(0, mx + 2):
                    k = kk(i, j)
                    Z[j, i] = u[k]
                    Z1[j, i] = 0.0
            X = np.linspace(x1, x2, mx + 2)
            Y = np.linspace(y1, y2, my + 2)
            A, B = np.meshgrid(Y, X)
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            surf1 = ax.plot_surface(A, B, Z, cmap='Greens', vmin=0.0, vmax=5.1, alpha=.4)
            surf2 = ax.plot_surface(A, B, Z1, color='b', vmin=0.0, vmax=5.1, alpha=1.0)
            plt.ioff()
            plt.show()

        print('aggregate convergence factor: ' + str((residuals[-1]/residuals[0])**(1.0/(len(residuals)-1.0))))
        print('residual reduction: ' + str(residuals[-1]/residuals[0]) + '\n')
        return u




