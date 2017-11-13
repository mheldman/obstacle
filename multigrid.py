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
        output += 'Fine grid size (' + str((self.levels[0].m + 2)**2) + ', ' + str((self.levels[0].m + 2)**2) + ')\n'
        output += str(self.levels[0].m**2) + ' fine grid unknowns\n'
        output += 'Coarse grid size (' + str((self.coarse_m + 2)**2) + ', ' + str((self.coarse_m + 2)**2) + ')\n'
        output += str(self.coarse_m ** 2) + ' coarse grid unknown(s)\n'
        return output


    def __init__(self, levels, coarse_m, smoother, coarse_solver=spsolve):

        self.levels = levels
        self.coarse_solver = coarse_solver
        self.coarse_m = coarse_m
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

    def __init__(self, m, A, R, P, x1, x2, y1, y2):

        if callable(A):
            self.A = A(m)
        else:
            self.A = A

        if callable(R):
            self.R = R(m)
        else:
            self.R = R

        if callable(P):
            self.P = P(m)
        else:
            self.P = P

        self.m = m
        self.hx = (x2 - x1) / (m + 1)
        self.hy = (y2 - y1) / (m + 1)

class linear_pfas_solver:
    '''
    Builds a projected full approximation scheme solver object for linear complementarity problems
    Input is a list of levels, which each contain a restriction and prolongation operator, and a coarse grid solver
    '''

    def __repr__(self):
        output = 'PFAS solver\n'
        output += 'Number of levels = ' + str(len(self.levels)) + '\n'
        output += 'Fine grid size (' + str((self.levels[0].m + 2)**2) + ', ' + str((self.levels[0].m + 2)**2) + ')\n'
        output += str(self.levels[0].m**2) + ' fine grid unknowns\n'
        output += 'Coarse grid size (' + str((self.coarse_m + 2)**2) + ', ' + str((self.coarse_m + 2)**2) + ')\n'
        output += str(self.coarse_m ** 2) + ' coarse grid unknown(s)\n'
        return output


    def __init__(self, levels, coarse_m, smoother, coarse_solver=pgs):

        self.levels = levels
        self.coarse_solver = coarse_solver
        self.coarse_m = coarse_m
        self.level = self.levels[0]
        self.smoother = smoother

    def lvl_solve(self, lvl, u, b, cycle):

        self.level = self.levels[lvl]
        print(lvl * "    " + "Grid level " + str(lvl) + ", m = " + str(self.level.m))
        A = self.level.A
        u = self.smoother(A, u, b)
        R = self.level.R
        coarse_u = R.dot(u)

        if lvl < len(self.levels) - 2:
            r = b - A.dot(u)
            coarse_b = R.dot(r)
            coarse_A = self.levels[lvl + 1].A
            coarse_b = coarse_b + coarse_A.dot(coarse_u)
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
            r = b - A.dot(u)
            coarse_b = R.dot(r)
            coarse_A = self.levels[lvl + 1].A
            coarse_b = coarse_b + coarse_A.dot(coarse_u)
            uold = np.zeros_like(coarse_u)
            du = 1.0
            while du > 10 ** -5:
                for i in range(0, len(uold)):
                    uold[i] = coarse_u[i]
                coarse_u = self.smoother(coarse_A, coarse_u, coarse_b)
                du = self.level.m * np.linalg.norm(uold - coarse_u)

        P = self.levels[lvl + 1].P
        coarse_u = coarse_u - R.dot(u)
        u += P.dot(coarse_u)
        self.level = self.levels[lvl]
        u = self.smoother(A, u, b)
        print(lvl * "    " + "Grid level " + str(lvl) + ", m = " + str(self.level.m))

    def solve(self, b, u0=None, cycle='FV'):

        if u0 is None:
            u0 = np.zeros_like(b)

        u = np.array(u0)
        self.lvl_solve(0, u, b, cycle)
        return u











def vcycle(m, vh, A, fh, eta1 = 3, eta2 = 3, numcycles = 5, cyclenum = 0):
    if cyclenum < numcycles:
        vh = gs(A, vh, fh)
        f2h = restrict_inj(m).dot(fh - A.dot(vh))
        m = int((m - 1) / 2)
        N = (m + 2) ** 2
        v2h = np.zeros(N)
        cyclenum += 1
        A = poisson2d(m)
        v2h = vcycle(m, v2h, A, f2h, eta1 = 3, eta2 = 3, numcycles = numcycles, cyclenum = cyclenum)
    else:
        vh = spsolve(A, fh)
        return vh
    vh = vh + interpolate(m).dot(v2h)
    m = 2 * m + 1
    A = poisson2d(m)
    vh = gs(A, vh, fh)
    return vh


def fmg(m, fh, eta0 = 1, eta1 = 3, eta2 = 3, numcycles = 5, cyclenum = 0):
    if cyclenum < numcycles:
            f2h = restrict_inj(m).dot(fh)
            cyclenum += 1
            m = int((m - 1) / 2)
            v2h = fmg(m, f2h, eta0=eta0, eta1=eta1, eta2=eta2, numcycles=numcycles, cyclenum=cyclenum)
            vh = interpolate(m).dot(v2h)
            m = 2 * m + 1
    else:
        vh = zeros((m + 2) ** 2)
        return vh
    A = poisson2d(m)
    vh = vcycle(m, vh, A, fh, eta1 = eta1, eta2 = eta2, numcycles = numcycles - cyclenum + 1)
    return vh

def pfas(m, uh, Ah, fh, eta=1, numcycles = 5, cyclenum = 0):
    if cyclenum < numcycles:
            vh = pgs(uh, Ah, fh, (m + 2) ** 2, maxiters=eta)
            r2h = restrict_inj(m).dot(fh - Ah.dot(vh))
            m = int((m - 1) / 2)
            v2h = np.transpose(interpolate(m)).dot(vh)
            v2h = np.maximum(v2h, 0.0)
            cyclenum += 1
            A2h, _, _ = poisson2d(m, bvals=True, a=-2.0, b=2.0)
            f2h = r2h + A2h.dot(v2h)
            u2h = pfas(m, v2h, A2h, f2h, eta=eta, numcycles=numcycles, cyclenum=cyclenum)
            e2h = u2h - v2h
    else:
        du = 1.0
        uold = zeros(len(uh))
        while du > 10**-5:
            for i in range(0, len(uh)):
                uold[i] = uh[i]
            uh = pgs(uh, Ah, fh, (m + 2)**2, maxiters=eta)
            du = m*np.linalg.norm(uold - uh)
        return uh
    vh = vh + interpolate(m).dot(e2h)
    m = 2 * m + 1
    vh = np.maximum(vh, 0.0)
    vh = pgs(vh, Ah, fh, (m + 2) ** 2, maxiters=eta)
    return vh

def enforce_bdry_conds(U, F, bvals):
    for k in bvals:
        U[k] = F[k]
    return U
''''''

