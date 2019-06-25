from __future__ import absolute_import
from obstacle import rspmethod_lcp_solver

__all__ = ['rs_smoother']

def rs_smoother(A, x, b, maxiters = 1, bvals=[]):
    rs_solver = rspmethod_lcp_solver(A, b, 1e-10, maxiters, bvals, ())
    return rs_solver.solve(0, init_iterate=x)


