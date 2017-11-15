import numpy as np
from numpy import zeros
from scipy import sparse
from scipy.sparse.linalg import spsolve
from GS import gs, pgs
from Poisson2D import poisson2d
from grid_transfers import *
from obstacle import obstaclersp

class multigrid_solver:
    '''
    Builds a multigrid solver object
    Input is a list of levels, which each contain a restriction and prolongation operator, and a coarse grid solver
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

        print(lvl * "    " + "Grid level " + str(lvl) + ", m = " + str(self.level.m))
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
            print((lvl + 1) * "    " + "Grid level " + str(lvl + 1) + ", m = " + str(self.coarse_m))
            coarse_u = self.coarse_solver(self.levels[-1].A, coarse_b)

        P = self.levels[lvl + 1].P
        u += P.dot(coarse_u)
        u = self.smoother(A, u, b)
        self.level = self.levels[lvl]
        print(lvl * "    " + "Grid level " + str(lvl) + ", m = " + str(self.level.m))

    def solve(self, b, u0=None, cycle='FV'):

        if u0 is None:
            u0 = np.zeros_like(b)

        u = np.array(u0)
        self.lvl_solve(0, u, b, cycle)
        return u


class level:

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
    Builds a projected full approximation scheme solver object for linear complementarity problems
    Input is a list of levels, which each contain a restriction and prolongation operator, and a coarse grid solver
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

    def lvl_solve(self, lvl, u, b, cycle):

        self.level = self.levels[lvl]
        print(lvl * "    " + "Grid level " + str(lvl) + ", mx = " + str(self.level.mx) + ", my = " + str(self.level.my))
        A = self.level.A
        u = self.smoother(A, u, b)
        R = self.level.R
        coarse_u = R.dot(u)
        r = b - A.dot(u)
        coarse_b = R.dot(r)
        coarse_A = self.levels[lvl + 1].A
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
            print((lvl + 1) * "    " + "Grid level " + str(lvl + 1) + ", mx = " + str(self.coarse_mx) + ", my = " + str(
                self.coarse_my))
            uold = np.zeros_like(coarse_u)
            du = 1.0
            while du > 10 ** -5:
                for i in range(0, len(uold)):
                    uold[i] = coarse_u[i]
                coarse_u = self.smoother(coarse_A, coarse_u, coarse_b)
                du = self.level.mx * np.linalg.norm(uold - coarse_u)

        P = self.levels[lvl + 1].P
        coarse_u = coarse_u - R.dot(u)
        u += P.dot(coarse_u)
        self.level = self.levels[lvl]
        u = self.smoother(A, u, b)
        print(lvl * "    " + "Grid level " + str(lvl) + ", mx = " + str(self.level.mx) + ", my = " + str(
            self.level.my))

    def solve(self, b, u0=None, cycle='FV'):

        if u0 is None:
            u0 = np.zeros_like(b)

        u = np.array(u0)
        self.lvl_solve(0, u, b, cycle)
        return u




