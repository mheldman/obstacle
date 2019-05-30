from __future__ import absolute_import
from obstacle.relaxation.relaxation import gauss_seidel, projected_gauss_seidel

__all__ = ['gs', 'pgs']

def gs(A, x, b, maxiters = 1):
    gauss_seidel(A, x, b, iterations=maxiters)

def pgs(A, x, b, maxiters = 1):
    projected_gauss_seidel(A, x, b, iterations=maxiters)
