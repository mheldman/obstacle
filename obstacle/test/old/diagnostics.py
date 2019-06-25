from scipy.sparse.linalg import eigs, inv
from Poisson2D import poisson2d, rhs
from grid_transfers import restrict_inj, restrict_fw, interpolate
from multigrid import linear_pfas_solver, level, multigrid_solver
from ReducedSpace import rspmethod_lcp_solver
from GS import pgs, gs
import numpy as np
from obstacle import box_obstacle_problem
from time import time
#from projected_newton import PNmethod_lcp_solver
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

def radialproblem_pfas_diagnostic(cycle, coarse_mx = 1, coarse_my = 1, min_num_levels = 6, max_num_levels = None,
                                                                                                  step = 1, smoothing_iters=1):
    Alpha = .68026
    Beta = .47152

    def psi(x, y):
        x = np.array([x])
        y = np.array([y])
        z = np.sqrt(np.maximum(1 - x ** 2 - y ** 2, 0.0))
        z[[z < 1 / np.sqrt(2)]] = -(x[[z < 1 / np.sqrt(2)]] ** 2 + y[[z < 1 / np.sqrt(2)]] ** 2) / np.sqrt(2) \
                                  + np.sqrt(2) - 1 / (2 * np.sqrt(2))
        return z[0]

    f = lambda x, y: 0.0
    g = lambda x, y: -Alpha * np.log(np.sqrt(x ** 2 + y ** 2)) + Beta
    x1, x2, y1, y2 = -2.0, 2.0, -2.0, 2.0
    bounds = (x1,x2,y1,y2)

    def uexact(x, y):
        r = np.sqrt(x ** 2 + y ** 2)
        cond1 = (r > .69797)
        cond2 = ~cond1
        Uexact = 0.*x
        Uexact[cond1] = g(x[cond1], y[cond1])
        Uexact[cond2] = psi(x[cond2], y[cond2])
        return Uexact

    if max_num_levels is None:
        max_num_levels = min_num_levels + 1
    plt.ion()
    timelist = []
    for i in range(min_num_levels, max_num_levels, step):
        levels = []
        num_levels = i
        mx, my = coarse_mx, coarse_my
        tstart = time()
        obstacle_problem = box_obstacle_problem(bounds, f, g, psi)
        obstacle_problem.discretize(mx, my)
        lvl = level(mx, my, poisson2d, restrict_inj, interpolate, x1, x2, y1, y2, bndry_pts=obstacle_problem.bndry_pts)  # currently uses only injection operator
        levels.append(lvl)
        for j in range(1, num_levels):
            mx = 2 * mx + 1
            my = 2 * my + 1
            obstacle_problem = box_obstacle_problem(bounds, f, g, psi)
            obstacle_problem.discretize(mx, my)
            lvl = level(mx, my, poisson2d, restrict_inj, interpolate, x1, x2, y1, y2, bndry_pts=obstacle_problem.bndry_pts)  # currently uses only injection operator
            levels.append(lvl)

        levels.reverse()

        pfas_solver = linear_pfas_solver(levels, coarse_mx, coarse_my, pgs, diagnostics=('show residuals', 'geometric convergence factor', 'show reduced space'))
        print('system setup time: ' + str(time() - tstart) + '\n')
        print(pfas_solver)
        tstart = time()
        U = obstacle_problem.solve(pfas_solver.solve, obstacle_problem.F, cycle=cycle, maxiters=100, smoothing_iters=smoothing_iters, accel=None)
        timex = time() - tstart
        timelist.append(timex)
        if len(timelist) > 1:
            print('time ratio: ' + str(timelist[-1]/timelist[-2]))
            print('average time ratio: ' + str((timelist[-1]/timelist[0])/(4.0**(len(timelist) - 1.0))))
        N = (mx + 2) * (my + 2)
        Uexact = np.zeros(N)
        X = np.linspace(x1, x2, mx + 2)
        Y = np.linspace(y1, y2, my + 2)
        if type(uexact(X, Y)) == float:
            kk = lambda i, j: j * (mx + 2) + i
            for j in range(0, my + 2):
                for i in range(0, mx + 2):
                    k = kk(i, j)
                    Uexact[k] = uexact(X[i], Y[j])
        else:
            [X, Y] = np.meshgrid(X, Y)
            Uexact = uexact(X.flatten(), Y.flatten())
        print('Error: ||U - Uexact||_inf = ' + str(np.linalg.norm(U - Uexact, np.inf)))
        print('time for ' + cycle + ': ' + str(timex) + '\n')

    '''
    plt.semilogy(Nlist, [timelist[i]/Nlist[i] for i in range(len(Nlist))], '*-')
    plt.xlabel('# unknowns (N)', fontsize=18)
    plt.ylabel('time/N', fontsize=18)
    plt.ioff()
    plt.show()
    '''
    return U, obstacle_problem

def radialproblem_rsp_diagnostic(coarse_mx = 1, coarse_my = 1, min_num_levels = 6, max_num_levels = None, step = 1):

    Alpha = .68026
    Beta = .47152

    def psi(x, y):
        x = np.array([x])
        y = np.array([y])
        z = np.sqrt(np.maximum(1 - x ** 2 - y ** 2, 0.0))
        z[[z < 1 / np.sqrt(2)]] = -(x[[z < 1 / np.sqrt(2)]] ** 2 + y[[z < 1 / np.sqrt(2)]] ** 2) / np.sqrt(2) \
                                  + np.sqrt(2) - 1 / (2 * np.sqrt(2))
        return z[0]

    f = lambda x, y: np.zeros(len(x))
    g = lambda x, y: -Alpha * np.log(np.sqrt(x ** 2 + y ** 2)) + Beta
    x1, x2, y1, y2 = -2.0, 2.0, -2.0, 2.0
    bounds = (x1,x2,y1,y2)
    def uexact(x, y):
        r = np.sqrt(x ** 2 + y ** 2)
        if r > .69797:
            return g(x, y)
        else:
            return psi(x, y)

    if max_num_levels is None:
        max_num_levels = min_num_levels + 1
    plt.ion()
    mx, my = coarse_mx, coarse_my
    for j in range(0, min_num_levels):
        mx = 2 * mx + 1
        my = 2 * my + 1

    for i in range(min_num_levels, max_num_levels, step):

        obstacle_problem = box_obstacle_problem(bounds, f, g, psi)
        obstacle_problem.discretize(mx, my)
        rsp_solver = rspmethod_lcp_solver(obstacle_problem.A, obstacle_problem.F, 1e-8, 100, diagnostics=('show reduced space'), bvals=obstacle_problem.bndry_pts)
        print(rsp_solver)
        tstart = time()
        U = obstacle_problem.solve(rsp_solver.solve, linear_solver = 'splu', bounds=bounds, mx = mx, my= my)
        timex = time() - tstart
        N = (mx + 2) * (my + 2)
        if uexact is not None:
            Uexact = np.zeros(N)
            X = np.linspace(x1, x2, mx + 2)
            Y = np.linspace(y1, y2, my + 2)
            if type(uexact(x,y)) == float:
                kk = lambda i, j: j * (mx + 2) + i
                for j in range(0, my + 2):
                    for i in range(0, mx + 2):
                        k = kk(i, j)
                        Uexact[k] = uexact(X[i], Y[j])
            else:
                [X, Y] = np.meshgrid(X, Y)
                Uexact = psi(X.flatten(), Y.flatten())
            print('Error: ||U - Uexact||_inf =', np.linalg.norm(U - Uexact, np.inf), '\n')
        print('\nTime for rsp:', timex, '\n')
        mx = 2*mx + 1
        my = 2*my + 1

def damproblem_pfas_diagnostic(cycle, coarse_mx = 1, coarse_my = 1, min_num_levels = 6, max_num_levels = None, step = 1, smoothing_iters=1):
    x1, x2, y1, y2 = 0.0, 16.0, 0.0, 24.0
    bounds = (x1, x2, y1, y2)
    c = 4.0

    f = lambda x, y: np.ones(len(x))
    psi = lambda x, y: np.zeros(len(x))

    def g(x, y):
        G = np.zeros(len(x))
        a = x2 - x1
        b = y2 - y1
        bc1 = (x == x1)
        G[bc1] = .5 * (b - y[bc1]) ** 2
        bc2 = (y <= c) & (x == x2)
        G[bc2] = .5 * (c - y[bc2]) ** 2
        bc3 = (y == y1)
        G[bc3] = ((a - x[bc3]) * b ** 2 + x[bc3] * c ** 2) / (2 * a)
        return G

    if max_num_levels is None:
        max_num_levels = min_num_levels + 1
    plt.ion()
    timelist = []
    Nlist = []
    for i in range(min_num_levels, max_num_levels, step):
        plt.close()
        levels = []
        num_levels = i
        tstart = time()
        mx, my = coarse_mx, coarse_my
        obstacle_problem = box_obstacle_problem(bounds, f, g, psi)
        obstacle_problem.discretize(mx, my)
        lvl = level(mx, my, poisson2d, restrict_inj, interpolate, x1, x2, y1, y2, bndry_pts=obstacle_problem.bndry_pts)  # currently uses only injection operator
        levels.append(lvl)
        for j in range(1, num_levels):
            mx = 2 * mx + 1
            my = 2 * my + 1
            obstacle_problem = box_obstacle_problem(bounds, f, g, psi)
            obstacle_problem.discretize(mx, my)
            lvl = level(mx, my, poisson2d(mx, my, x1, x2, y1, y2), restrict_inj(mx, my), interpolate(mx, my), x1, x2, y1, y2, bndry_pts=obstacle_problem.bndry_pts)  # currently uses only injection operator
            levels.append(lvl)

        levels.reverse()

        pfas_solver = linear_pfas_solver(levels, coarse_mx, coarse_my, pgs, diagnostics=('show residuals', 'geometric convergence factor'))
        print('system setup time: ' + str(time() - tstart) + '\n')
        print(pfas_solver)
        tstart = time()
        U = obstacle_problem.solve(pfas_solver.solve, obstacle_problem.F, cycle=cycle, maxiters=50, smoothing_iters=smoothing_iters)
        timex = time() - tstart
        timelist.append(timex)
        if len(timelist) > 1:
            print('time ratio: ' + str(timelist[-1] / timelist[-2]))
            print('average time ratio: ' + str((timelist[-1] / timelist[0]) / (4.0 ** (len(timelist) - 1.0))))
        Nlist.append((mx + 2)*(my + 2))
        print('time for ' + cycle + ': ' +  str(timex) + '\n')
    #plt.plot(Nlist, [timelist[i]/Nlist[i] for i in range(len(Nlist))], '*-')
    #plt.xlabel('# unknowns (N)', fontsize=18)
    #plt.ylabel('time/N', fontsize=18)
    #plt.ioff()
    #plt.show()

    return U, obstacle_problem


def damproblem_rsp_diagnostic(coarse_mx = 1, coarse_my = 1, min_num_levels = 6, max_num_levels = None, step = 1):
    x1, x2, y1, y2 = 0.0, 16.0, 0.0, 24.0
    bounds = (x1, x2, y1, y2)
    c = 4.0

    f = lambda x, y: 1.0 * x
    psi = lambda x, y: 0.0 * x

    def g(x, y):
        a = x2 - x1
        b = y2 - y1
        if x == x1:
            return .5 * (b - y) ** 2
        elif y <= c and x == x2:
            return .5 * (c - y) ** 2
        elif y == y1:
            return ((a - x) * b ** 2 + x * c ** 2) / (2 * a)
        else:
            return 0.0

    if max_num_levels is None:
        max_num_levels = min_num_levels + 1
    plt.ion()
    mx, my = coarse_mx, coarse_my
    for j in range(0, min_num_levels):
        mx = 2 * mx + 1
        my = 2 * my + 1

    for i in range(min_num_levels, max_num_levels, step):

        obstacle_problem = box_obstacle_problem(bounds, f, g, psi)
        obstacle_problem.discretize(mx, my)
        rsp_solver = rspmethod_lcp_solver(obstacle_problem.A, obstacle_problem.F, 1e-8, 100, diagnostics=('show reduced space'), bvals=obstacle_problem.bndry_pts)
        print(rsp_solver)
        tstart = time()
        U = obstacle_problem.solve(rsp_solver.solve, linear_solver = 'splu', bounds=(x1,x2,y1,y2), mx = mx, my=my)
        timex = time() - tstart
        print('\nTime for rsp:', timex, '\n')
        mx = 2*mx + 1
        my = 2*my + 1
    return rsp_solver.iterates

def unconstrproblem_pfas_diagnostic(cycle, coarse_mx = 1, coarse_my = 1, min_num_levels = 6, max_num_levels = None,
                                                                                        step = 1):
    #f = lambda x, y: -8.0 * (np.pi ** 2) * np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
    f = lambda x, y: 12 * (y ** 4 - 1) * x ** 2 + 12 * (x ** 4 - 1) * y ** 2
    g = lambda x, y: 0.0 * x
    obstacle = 1.0
    psi = lambda x, y: -obstacle

    x1, x2, y1, y2 = -1.0, 1.0, -1.0, 1.0
    bounds = (x1, x2, y1, y2)
    uexact = lambda x, y: np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)

    if max_num_levels is None:
        max_num_levels = min_num_levels + 1
    plt.ion()
    for i in range(min_num_levels, max_num_levels, step):
        levels = []
        num_levels = i
        mx, my = coarse_mx, coarse_my
        lvl = level(mx, my, poisson2d, restrict_inj, interpolate, x1, x2, y1, y2)  # currently uses only injection operator
        levels.append(lvl)
        for j in range(1, num_levels):
            mx = 2 * mx + 1
            my = 2 * my + 1
            lvl = level(mx, my, poisson2d, restrict_inj, interpolate, x1, x2, y1, y2)  # currently uses only injection operator
            levels.append(lvl)
        N = (mx + 2)*(my + 2)
        levels.reverse()
        obstacle_problem = box_obstacle_problem(bounds, f, g, psi)
        obstacle_problem.discretize(mx, my)
        pfas_solver = linear_pfas_solver(levels, coarse_mx, coarse_my, pgs, diagnostics=('show residuals', 'geometric convergence factor', 'show reduced space'))

        print(pfas_solver)
        tstart = time()
        U = obstacle_problem.solve(pfas_solver.solve, obstacle_problem.F, tol=1e-10, cycle=cycle, maxiters=50, u0=np.zeros_like(obstacle_problem.F)+obstacle)
        timex = time() - tstart
        Uexact = spsolve(obstacle_problem.A, obstacle_problem.F) + obstacle_problem.P
        print('Error: ||U - Uexact||_inf =', np.linalg.norm(U - Uexact, np.inf), '\n')
        print('\nTime for ' + cycle + ':', timex, '\n')
        plt.semilogy(range(len(pfas_solver.residuals)), pfas_solver.residuals)
        plt.xlabel('iteration', fontsize=18)
        plt.ylabel('residual', fontsize=18)
        plt.savefig('residual_plot('+str(mx)+','+str(my)+').png')
        plt.show()

def unconstrproblem_multigrid_diagnostic(cycle, coarse_mx = 1, coarse_my = 1, min_num_levels = 6, max_num_levels = None,
                                                                                                            step = 1):
    #f = lambda x, y: -8.0 * (np.pi ** 2) * np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
    f = lambda x, y: 12*(y**4 - 1)*x**2 + 12*(x**4 - 1)*y**2
    g = lambda x, y: 0.0*x
    psi = lambda x, y: -10**10

    x1, x2, y1, y2 = -1.0, 1.0, -1.0, 1.0
    bounds = (x1, x2, y1, y2)
    uexact = lambda x, y: np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)

    if max_num_levels is None:
        max_num_levels = min_num_levels + 1
    plt.ion()
    for i in range(min_num_levels, max_num_levels, step):
        levels = []
        num_levels = i
        mx, my = coarse_mx, coarse_my
        lvl = level(mx, my, poisson2d, restrict_inj, interpolate, x1, x2, y1, y2)  # currently uses only injection operator
        levels.append(lvl)
        for j in range(1, num_levels):
            mx = 2 * mx + 1
            my = 2 * my + 1
            lvl = level(mx, my, poisson2d, restrict_inj, interpolate, x1, x2, y1, y2)  # currently uses only injection operator
            levels.append(lvl)
        N = (mx + 2)*(my + 2)
        levels.reverse()
        obstacle_problem = box_obstacle_problem(bounds, f, g, psi)
        obstacle_problem.discretize(mx, my)
        pfas_solver = multigrid_solver(levels, coarse_mx, coarse_my, gs, coarse_solver=spsolve)
        print(pfas_solver)
        tstart = time()
        U = obstacle_problem.solve(pfas_solver.solve, obstacle_problem.F, tol=1e-10, cycle=cycle, maxiters=50,u0=np.zeros_like(obstacle_problem.F)+10**10)
        timex = time() - tstart
        Uexact = spsolve(obstacle_problem.A, obstacle_problem.F) + obstacle_problem.P
        print('Error: ||U - Uexact||_inf =', np.linalg.norm(U - Uexact, np.inf), '\n')
        print('\nTime for ' + cycle + ':', timex, '\n')
        plt.semilogy(range(len(pfas_solver.residuals)), pfas_solver.residuals)
        plt.xlabel('iteration', fontsize=18)
        plt.ylabel('residual', fontsize=18)
        plt.savefig('residual_plot('+str(mx)+','+str(my)+').png')
        plt.show()

def radialproblem_PN_diagnostic(coarse_mx = 1, coarse_my = 1, min_num_levels = 6, max_num_levels = None, step = 1):

    Alpha = .68026
    Beta = .47152

    def psi(x, y):
        x = np.array([x])
        y = np.array([y])
        z = np.sqrt(np.maximum(1 - x ** 2 - y ** 2, 0.0))
        z[[z < 1 / np.sqrt(2)]] = -(x[[z < 1 / np.sqrt(2)]] ** 2 + y[[z < 1 / np.sqrt(2)]] ** 2) / np.sqrt(2) \
                                  + np.sqrt(2) - 1 / (2 * np.sqrt(2))
        return z[0]

    f = lambda x, y: 0.0
    g = lambda x, y: -Alpha * np.log(np.sqrt(x ** 2 + y ** 2)) + Beta
    x1, x2, y1, y2 = -2.0, 2.0, -2.0, 2.0
    bounds = (x1,x2,y1,y2)
    def uexact(x, y):
        r = np.sqrt(x ** 2 + y ** 2)
        if r > .69797:
            return g(x, y)
        else:
            return psi(x, y)

    if max_num_levels is None:
        max_num_levels = min_num_levels + 1
    plt.ion()
    mx, my = coarse_mx, coarse_my
    for j in range(0, min_num_levels):
        mx = 2 * mx + 1
        my = 2 * my + 1

    for i in range(min_num_levels, max_num_levels, step):


        obstacle_problem = box_obstacle_problem(bounds, f, g, psi)
        obstacle_problem.discretize(mx, my)
        PN_solver = PNmethod_lcp_solver(obstacle_problem.A, obstacle_problem.F, 1e-8, 100, diagnostics=('show reduced space'), bvals=obstacle_problem.bndry_pts)
        print(PN_solver)
        tstart = time()
        U = obstacle_problem.solve(PN_solver.solve, linear_solver = 'splu', bounds=bounds, mx = mx, my= my)
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
        print('\nTime for PN:', timex, '\n')
        mx = 2*mx + 1
        my = 2*my + 1

def damproblem_PN_diagnostic(coarse_mx = 1, coarse_my = 1, min_num_levels = 6, max_num_levels = None, step = 1):
    x1, x2, y1, y2 = 0.0, 16.0, 0.0, 24.0
    bounds = (x1, x2, y1, y2)
    c = 4.0

    f = lambda x, y: 1.0 * x
    psi = lambda x, y: 0.0 * x

    def g(x, y):
        a = x2 - x1
        b = y2 - y1
        if x == x1:
            return .5 * (b - y) ** 2
        elif y <= c and x == x2:
            return .5 * (c - y) ** 2
        elif y == y1:
            return ((a - x) * b ** 2 + x * c ** 2) / (2 * a)
        else:
            return 0.0

    if max_num_levels is None:
        max_num_levels = min_num_levels + 1
    plt.ion()
    mx, my = coarse_mx, coarse_my
    for j in range(0, min_num_levels):
        mx = 2 * mx + 1
        my = 2 * my + 1

    for i in range(min_num_levels, max_num_levels, step):

        obstacle_problem = box_obstacle_problem(bounds, f, g, psi)
        obstacle_problem.discretize(mx, my)
        PN_solver = PNmethod_lcp_solver(obstacle_problem.A, obstacle_problem.F, 1e-8, 100, diagnostics=('show reduced space'), bvals=obstacle_problem.bndry_pts)
        print(PN_solver)
        tstart = time()
        U = obstacle_problem.solve(PN_solver.solve, linear_solver = 'splu', bounds=(x1,x2,y1,y2), mx = mx, my=my)
        timex = time() - tstart
        print('\nTime for PN:', timex, '\n')
        mx = 2*mx + 1
        my = 2*my + 1
    return PN_solver.iterates

def radialproblem_pgs_diagnostic(coarse_mx = 1, coarse_my = 1, min_num_levels = 6, max_num_levels = None, step = 1):

    Alpha = .68026
    Beta = .47152

    def psi(x, y):
        x = np.array([x])
        y = np.array([y])
        z = np.sqrt(np.maximum(1 - x ** 2 - y ** 2, 0.0))
        z[[z < 1 / np.sqrt(2)]] = -(x[[z < 1 / np.sqrt(2)]] ** 2 + y[[z < 1 / np.sqrt(2)]] ** 2) / np.sqrt(2) \
                                  + np.sqrt(2) - 1 / (2 * np.sqrt(2))
        return z[0]

    f = lambda x, y: 0.0
    g = lambda x, y: -Alpha * np.log(np.sqrt(x ** 2 + y ** 2)) + Beta
    x1, x2, y1, y2 = -2.0, 2.0, -2.0, 2.0
    bounds = (x1, x2, y1, y2)

    def uexact(x, y):
        r = np.sqrt(x ** 2 + y ** 2)
        if r > .69797:
            return g(x, y)
        else:
            return psi(x, y)

    if max_num_levels is None:
        max_num_levels = min_num_levels + 1
    plt.ion()
    mx, my = coarse_mx, coarse_my
    for j in range(0, min_num_levels):
        mx = 2 * mx + 1
        my = 2 * my + 1

    for i in range(min_num_levels, max_num_levels, step):

        obstacle_problem = box_obstacle_problem(bounds, f, g, psi)
        obstacle_problem.discretize(mx, my)
        tstart = time()
        U = obstacle_problem.solve(pgs, obstacle_problem.A, obstacle_problem.U, obstacle_problem.F, maxiters=100, plot_active=True, bounds=(x1,x2,y1,y2), mx = mx, my=my)
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
        print('\nTime for pgs:', timex, '\n')
        mx = 2 * mx + 1
        my = 2 * my + 1

def damproblem_pgs_diagnostic(coarse_mx = 1, coarse_my = 1, min_num_levels = 6, max_num_levels = None, step = 1):
    x1, x2, y1, y2 = 0.0, 16.0, 0.0, 24.0
    bounds = (x1, x2, y1, y2)
    c = 4.0

    f = lambda x, y: np.ones(len(x))
    psi = lambda x, y: np.zeros(len(x))

    def g(x, y):
        G = np.zeros(len(x))
        a = x2 - x1
        b = y2 - y1
        bc1 = (x == x1)
        G[bc1] = .5 * (b - y[bc1]) ** 2
        bc2 = (y <= c) & (x == x2)
        G[bc2] = .5 * (c - y[bc2]) ** 2
        bc3 = (y == y1)
        G[bc3] = ((a - x[bc3]) * b ** 2 + x[bc3] * c ** 2) / (2 * a)
        return G

    if max_num_levels is None:
        max_num_levels = min_num_levels + 1
    plt.ion()
    mx, my = coarse_mx, coarse_my
    for j in range(0, min_num_levels):
        mx = 2 * mx + 1
        my = 2 * my + 1

    for i in range(min_num_levels, max_num_levels, step):

        obstacle_problem = box_obstacle_problem(bounds, f, g, psi)
        obstacle_problem.discretize(mx, my)
        tstart = time()
        U = obstacle_problem.solve(pgs,  obstacle_problem.A, obstacle_problem.U, obstacle_problem.F, maxiters=100, plot_active=True, bounds=(x1,x2,y1,y2), mx = mx, my=my)
        timex = time() - tstart
        print('\nTime for pgs:', timex, '\n')
        mx = 2*mx + 1
        my = 2*my + 1

def elastictorsion_pfas_diagnostic(cycle, coarse_mx = 1, coarse_my = 1, min_num_levels = 6, max_num_levels = None,

                                                                                               step = 1, smoothing_iters=1):
    '''
    mx = coarse_mx
    my = coarse_my
    num_cycles = 7
    for i in range(num_cycles):
        mx = 2 * mx + 1
        my = 2 * my + 1
    N = (mx + 2) * (my + 2)
    '''
    x1, x2, y1, y2 = 0.0, 1.0, 0.0, 1.0
    w = np.linspace(x1, x2, 1000)
    # w_list = [w for i in range(N)]
    # w = np.array(w_list)

    psi = lambda x, y:  -np.minimum(np.minimum(x, y), np.minimum(1.0-x, 1.0-y))
    f = lambda x, y: 8.0*np.ones(len(y))
    g = lambda x, y: 0.*y
    bounds = (x1, x2, y1, y2)

    if max_num_levels is None:
        max_num_levels = min_num_levels + 1
    plt.ion()
    timelist=[]
    for i in range(min_num_levels, max_num_levels, step):
        levels = []
        num_levels = i
        tstart = time()
        mx, my = coarse_mx, coarse_my
        obstacle_problem = box_obstacle_problem(bounds, f, g, psi)
        obstacle_problem.discretize(mx, my)
        lvl = level(mx, my, poisson2d, restrict_inj, interpolate, x1, x2, y1, y2, bndry_pts=obstacle_problem.bndry_pts)  # currently uses only injection operator
        levels.append(lvl)
        for j in range(1, num_levels):
            mx = 2 * mx + 1
            my = 2 * my + 1
            obstacle_problem = box_obstacle_problem(bounds, f, g, psi)
            obstacle_problem.discretize(mx, my)
            lvl = level(mx, my, poisson2d, restrict_inj, interpolate, x1, x2, y1, y2, bndry_pts=obstacle_problem.bndry_pts)  # currently uses only injection operator
            levels.append(lvl)

        levels.reverse()

        pfas_solver = linear_pfas_solver(levels, coarse_mx, coarse_my, pgs, diagnostics=('show residuals', 'geometric convergence factor'))
        print('system setup time: ' + str(time() - tstart) + '\n')
        print(pfas_solver)
        tstart = time()
        U = obstacle_problem.solve(pfas_solver.solve, obstacle_problem.F, cycle=cycle, maxiters=100, smoothing_iters=smoothing_iters)
        timex = time() - tstart
        timelist.append(timex)
        if len(timelist) > 1:
            print('time ratio: ' + str(timelist[-1]/timelist[-2]))
            print('average time ratio: ' + str((timelist[-1]/timelist[0])/(4.0**(len(timelist) - 1.0))))
        N = (mx + 2) * (my + 2)
        uexact=None
        if uexact is not None:
            Uexact = np.zeros(N)
            X = np.linspace(x1, x2, mx + 2)
            Y = np.linspace(y1, y2, my + 2)
            kk = lambda i, j: j * (mx + 2) + i
            for j in range(0, my + 2):
                for i in range(0, mx + 2):
                    k = kk(i, j)
                    Uexact[k] = uexact(X[i], Y[j])
                print('Error: ||U - Uexact||_inf = ' + str(np.linalg.norm(U - Uexact, np.inf)))
        print('time for ' + cycle + ': ' + str(timex) + '\n')
    '''
    plt.semilogy(Nlist, [timelist[i]/Nlist[i] for i in range(len(Nlist))], '*-')
    plt.xlabel('# unknowns (N)', fontsize=18)
    plt.ylabel('time/N', fontsize=18)
    plt.ioff()
    plt.show()
    '''
    return U, obstacle_problem

def elastictorsion_rsp_diagnostic(coarse_mx = 1, coarse_my = 1, min_num_levels = 6, max_num_levels = None, step = 1):

    x1, x2, y1, y2 = 0.0, 1.0, 0.0, 1.0
    w = np.linspace(x1, x2, 1000)
    # w_list = [w for i in range(N)]
    # w = np.array(w_list)

    psi = lambda x, y: -np.minimum(x, y, 1.0-x, 1.0-y)
    f = lambda x, y: 8.0
    g = lambda x, y: 0.0
    bounds = (x1, x2, y1, y2)
    if max_num_levels is None:
        max_num_levels = min_num_levels + 1
    plt.ion()
    mx, my = coarse_mx, coarse_my
    for j in range(0, min_num_levels):
        mx = 2 * mx + 1
        my = 2 * my + 1

    for i in range(min_num_levels, max_num_levels, step):

        obstacle_problem = box_obstacle_problem(bounds, f, g, psi)
        obstacle_problem.discretize(mx, my)
        rsp_solver = rspmethod_lcp_solver(obstacle_problem.A, obstacle_problem.F, 1e-8, 100, bvals=obstacle_problem.bndry_pts)
        print(rsp_solver)
        tstart = time()
        U = obstacle_problem.solve(rsp_solver.solve, linear_solver = 'splu', bounds=(x1,x2,y1,y2), mx = mx, my=my, smoothing_iters=smoothing_iters)
        timex = time() - tstart
        print('\nTime for rsp:', timex, '\n')
        mx = 2*mx + 1
        my = 2*my + 1
    return rsp_solver.iterates




