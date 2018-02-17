import numpy as np
from scipy.sparse.linalg import spsolve
from GS import pgs

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


    def __init__(self, levels, coarse_mx, coarse_my, smoother, coarse_solver=spsolve):

        '''
        :param levels: A list of level objects
        :param coarse_mx:
        :param coarse_my:
        :param smoother:
        :param coarse_solver:
        '''

        self.levels = levels
        self.coarse_solver = coarse_solver
        self.coarse_mx = coarse_mx
        self.coarse_my = coarse_my
        self.level = self.levels[0]
        self.smoother = smoother

    def lvl_solve(self, lvl, u, b, cycle):

        self.level = self.levels[lvl]
        A = self.level.A
        R = self.level.R

        print(lvl * "    " + "Grid " + str(lvl) + ", mx = " + str(self.level.mx) + ", my = " + str(self.level.my))
        u = self.smoother(A, u, b)
        r = b - A.dot(u)
        coarse_b = R.dot(r)

        if lvl < len(self.levels) - 2:
            coarse_u = np.zeros_like(coarse_b)
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
            print((lvl + 1) * "    " + "Grid " + str(lvl + 1) + ", mx = " + str(self.coarse_mx) + ", my = " + str(
                self.coarse_my))
            coarse_u = self.coarse_solver(self.levels[-1].A, coarse_b)

        P = self.levels[lvl + 1].P
        u += P.dot(coarse_u)
        u = self.smoother(A, u, b)
        self.level = self.levels[lvl]
        print(lvl * "    " + "Grid " + str(lvl) + ", mx = " + str(self.level.mx) + ", my = " + str(
            self.level.my))

    def solve(self, b, u0=None, cycle='FV'):

        if u0 is None:
            u0 = np.zeros_like(b)

        u = np.array(u0)
        self.lvl_solve(0, u, b, cycle)
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

    def __init__(self, mx, my, A, R, P, x1, x2, y1, y2):

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

        self.mx = mx
        self.my = my
        self.hx = (x2 - x1) / (mx + 1)
        self.hy = (y2 - y1) / (my + 1)

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


    def __init__(self, levels, coarse_mx, coarse_my, smoother, coarse_solver=pgs):

        self.levels = levels
        self.coarse_solver = coarse_solver
        self.coarse_mx = coarse_mx
        self.coarse_my = coarse_my
        self.level = self.levels[0]
        self.smoother = smoother
        self.mu = .15

    def lvl_solve(self, lvl, u, b, cycle):

        self.level = self.levels[lvl]
        print(lvl * "    " + "Grid " + str(lvl) + ", mx = " + str(self.level.mx) + ", my = " + str(self.level.my))
        A = self.level.A
        self.smoother(A, u, b)
        R = self.level.R
        coarse_u = R.dot(u)
        coarse_b = R.dot(b - A.dot(u))
        coarse_A = self.levels[lvl + 1].A #needs at least two levels
        coarse_b = coarse_b + coarse_A.dot(coarse_u)
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
            print((lvl + 1) * "    " + "Grid " + str(lvl + 1) + ", mx = " + str(self.coarse_mx) + ", my = " + str(
                self.coarse_my))
            uold = np.zeros_like(coarse_u)
            du = 1.0
            while du > 10 ** -12:
                for i in range(0, len(uold)):
                    uold[i] = coarse_u[i]
                self.smoother(coarse_A, coarse_u, coarse_b)
                du = (self.coarse_mx + 1) * np.linalg.norm(uold - coarse_u)

        P = self.levels[lvl + 1].P
        u += P.dot(coarse_u - R.dot(u))
        self.level = self.levels[lvl]
        self.smoother(A, u, b)
        print(lvl * "    " + "Grid " + str(lvl) + ", mx = " + str(self.level.mx) + ", my = " + str(
            self.level.my))

    def solve(self, b, u0=None, cycle='FV'):

        if u0 is None:
            u0 = np.zeros_like(b)

        u = np.array(u0)
        self.lvl_solve(0, u, b, cycle)
        return u




