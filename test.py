from Poisson2D import poisson2d, rhs
from grid_transfers import restrict_inj, restrict_fw, interpolate
from multigrid import linear_pfas_solver, level, multigrid_solver
from GS import pgs, gs
import numpy as np
from obstacle import box_obstacle_problem
from time import time
from ReducedSpace import rspmethod_lcp_solver



def multigrid_test(bounds, f, g, cycle, uexact = None, coarse_mx = 1,\
                                coarse_my = 1, min_num_levels = 5, max_num_levels = None):
    '''
    Test for multigrid solver on Poisson equation (1)

            u_xx + u_yy = f   on Omega
            u(x, y) = g(x, y) on boundary Omega.

    This solver restrict Omega to the box region [x1, x2] x [y1, y2].

    Multigrid is an optimal iterative linear solver which solves a discretized PDE
    recursively on successively coarser grids to produce improved initial iterates
    for the fine grid problem.

    Parameters:

    bounds {tuple}:
    (x1, x2, y1, y2)
    Defines the region [x1, x2] x [y1, y2]

    f {callable}: rhs of u_xx + u_yy = f

    g {callable}:  boundary condition for solution u

    cycle {str}: cycling scheme for PFAS
    options: 'V' - v-cycle, 'W' - w-cycle, 'FV' - standard F-cycle
    'FW' - F-cycle + W-cycle, 'fmgV' - F-cycle beginning on coarsest grid
    (interpolates rhs down to coarse grid -- should start on coarse grid)

    Optional parameters:

    uexact {callable}: exact solution. Default is None.

    coarse_mx {int}: horizontal grid spacing on coarsest PFAS grid. Default is coarse_mx = 1.

    coarse_my {int}: vertical grid spacing on coarsest PFAS grid. Default is coarse_my = 1.

    min_num_levels, max_num_levels {int}: Solve problem max_num_levels - min_num_levels
    times on increasingly finer fine grids. Default is max_num_levels = min_num_levels + 1.

    Output:

    U {np.array}: Discrete fine grid solution of .


    '''

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
        solver = multigrid_solver(levels, coarse_mx, coarse_my, gs)
        print(solver)
        F = rhs(f, mx, my = my, g=g, x1=x1,x2=x2, y1=y1, y2=y2)
        U = solver.solve(F, cycle=cycle)
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
Defines the region Omega from (2) as Omega = [x1, x2] x [y1, y2]

f {callable}: rhs of (2b)

g {callable}  rhs of (2d)

psi {callable}: obstacle -- rhs of (2a)

cycle {str}: cycling scheme for PFAS
options: 'V' - v-cycle, 'W' - w-cycle, 'FV' - standard F-cycle
'FW' - F-cycle + W-cycle, 'fmgV' - F-cycle beginning on coarsest grid 
(interpolates rhs down to coarse grid -- should start on coarse grid)

uexact {callable}: exact solution of (2)

coarse_mx {int}: horizontal grid spacing on coarsest PFAS grid

coarse_my {int}: vertical grid spacing on coarsest PFAS grid

min_num_levels, max_num_levels {int}: Solve problem max_num_levels - min_num_levels
times on increasingly finer fine grids. Default is max_num_levels = min_num_levels + 1.

Output:

U {np.array}: Discrete fine grid solution vector of (2).
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
            lvl = level(mx, my, poisson2d, restrict_inj, interpolate, x1, x2, y1, y2) #currently uses only injection operator
            levels.append(lvl)
        levels.reverse()

        obstacle_problem = box_obstacle_problem(bounds, f, g, psi)
        obstacle_problem.discretize(mx, my)
        pfas_solver = linear_pfas_solver(levels, coarse_mx, coarse_my, pgs)
        print(pfas_solver)
        for i in range(1):
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
'''
def fixed_point_test(uexact, solver, **kwargs):

    U = uexact
    return U == solver(uexact, **kwargs)

def pfas_fixed_point_test(bounds, f, g, psi, cycle, uexact = None, coarse_mx = 1,\
                                coarse_my = 1, min_num_levels = 5, max_num_levels = None):

'''

def rsp_test(bounds, f, g, psi, mx, my, linear_solver, tol = 10**-8, maxiters = 100, uexact = None):

    obstacle_p = box_obstacle_problem(bounds, f, g, psi)
    obstacle_p.discretize(mx, my)
    rsp_solver = rspmethod_lcp_solver(obstacle_p.A, obstacle_p.F, maxiters = maxiters,\
                                  tol = tol, fixed_vals=obstacle_p.bndry_pts)

    tstart = time()
    U = obstacle_p.solve(rsp_solver.solve, linear_solver = linear_solver)
    timex = time() - tstart

    N = (mx + 2) * (my + 2)
    x1, x2, y1, y2 = bounds
    if uexact is not None:
        Uexact = np.zeros(N)
        X = np.linspace(x1, x2, mx + 2)
        Y = np.linspace(y1, y2, my + 2)
        kk = lambda i, j: j * (mx + 2) + i
        for j in range(0, my + 2):
            for i in range(0, mx + 2):
                k = kk(i, j)
                Uexact[k] = uexact(X[i], Y[j])
        #U = obstacle_p.solve(rsp_solver.solve, init_iterate=Uexact, linear_solver=linear_solver)
        print('Error: ||U - Uexact||_inf =', np.linalg.norm(U - Uexact, np.inf), '\n')
    print('\nTime for rsp:', timex, '\n')
    print('Number of iterations =', rsp_solver.num_iterations, '\n')
    return U
