from Poisson2D import poisson2d
from grid_transfers import restrict_inj, restrict_fw, interpolate
from multigrid import linear_pfas_solver, level
from GS import pgs
import numpy as np
from obstacle import box_obstacle_problem
from time import time

"""
Test for obstacle problem solver using PFAS. 
PFAS solves linear complementarity problems of the form (1):

u(x) >= 0               x in Omega
Lu(x) <= f(x)           x in Omega
u(x)[f(x) - Lu(x)] = 0  x in Omega
u(x) = g(x)             x on the boundary of Omega,

where L is a second-order elliptic operator.
This implementation restricts Omega to the square region [x1, x2] x [y1, y2].

A closely related problem is the obstacle problem (2):

u(x) >= psi(x)          x in Omega                    (2a)
Lu(x) <= f(x)           x in Omega                    (2b)
u(x)[f(x) - Lu(x)] = 0  x in Omega                    (2c)
u(x) = g(x)             x on the boundary of Omega,   (2d)

which can be put in form (1) via the transformation (3)

v(x) := u - psi
h(x) := f - Lu(x)
k(x) := g - psi

v(x) >= 0               x in Omega
Lv(x) <= h(x)           x in Omega
v(x)[h(x) - Lv(x)] = 0  x in Omega
v(x) = k(x)             x on the boundary of Omega.

This program solves the obstacle problem (2) on Omega = [x1, x2] x [y1, y2] by solving
the transformed problem (3) via PFAS.

Parameters:

bounds {tuple}: 
(x1, x2, y1, y2)
Rectangular grid on which the problem is defined

f {callable}: rhs of (2b)

g {callable}  rhs of (2d)

psi {callable}: obstacle -- rhs of (2a)

cycle {str}: cycling scheme for PFAS
options: 'V' - v-cycle, 'W' - w-cycle, 'FV' - standard F-cycle
'FW' - F-cycle + W-cycle, 'fmgV' - F-cycle beginning on finest grid

uexact {callable}: exact solution of (2)

coarse_mx {int}: horizontal grid spacing on coarsest PFAS grid

coarse_my {int}: vertical grid spacing on coarsest PFAS grid

min_num_levels, max_num_levels {int}: Solve problem max_num_levels - min_num_levels
times on increasingly finer fine grids. Default is max_num_levels = min_num_levels + 1.
"""

def pfas_test(bounds, f, g, psi, cycle, uexact = None, coarse_mx = 1,\
                                coarse_my = 1, min_num_levels = 5, max_num_levels = None):

    x1, x2, y1, y2 = bounds
    if max_num_levels is None:
        max_num_levels = min_num_levels + 1
    for i in range(min_num_levels, max_num_levels):
        levels = []
        num_levels = i
        tstart = time()
        mx, my = coarse_mx, coarse_my
        for j in range(num_levels):
            if j != 0:
                mx = 2 * mx + 1
                my = 2 * my + 1
            lvl = level(mx, my, poisson2d, restrict_inj, interpolate, x1, x2, y1, y2)
            levels.append(lvl)
        levels.reverse()
        obstacle_problem = box_obstacle_problem(bounds, f, g, psi)
        obstacle_problem.discretize(mx, my)
        pfas_solver = linear_pfas_solver(levels, coarse_mx, coarse_my, pgs)
        print(pfas_solver)
        U = obstacle_problem.solve(pfas_solver.solve, obstacle_problem.F, obstacle_problem.U, cycle)
        timex = time() - tstart
        N = (mx + 2) * (my + 2)
        if uexact is not None:
            Uexact = np.zeros(N)
            X = np.linspace(x1, x2, mx + 2)
            Y = np.linspace(y1, y2, my + 2)
            kk = lambda i, j: j * (mx + 2) + i
            for j in range(0, my + 2):
                for i in range(0, mx + 2):
                    k = kk(i, j)
                    Uexact[k] = uexact(X[i], Y[j])
            print('Error: ||U - Uexact||_inf =', np.linalg.norm(U - Uexact, np.inf), '\n')
        print('\nTime for ' + cycle + ':', timex, '\n')
        return U

