from __future__ import absolute_import
from obstacle.relaxation.relaxation import gauss_seidel, projected_gauss_seidel

__all__ = ['gs', 'pgs']

def gs(A, x, b, maxiters = 1, bvals=[]):
    return gauss_seidel(A, x, b, iterations=maxiters)

def pgs(A, x, b, psi=None, maxiters = 1, bvals=[], sweep='forward'):
    return projected_gauss_seidel(A, x, b, psi=psi, iterations=maxiters, sweep=sweep)


def pgs_obstacle(A, x, b, psi, maxiters = 1):
  x -= psi
  projected_gauss_seidel(A, x, b - A.dot(psi), iterations=maxiters, sweep='forward')
  x += psi
  return x
