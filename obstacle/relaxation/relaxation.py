"""Relaxation methods for linear systems."""

from warnings import warn

import numpy as np
from scipy import sparse
import obstacle.pfas_core as pfas_core

__all__ = ['gauss_seidel', 'projected_gauss_seidel']

def make_system(A, x, b, formats=None):

    if formats is None:
        pass
    elif formats == ['csr']:
        if sparse.isspmatrix_csr(A):
            pass
        elif sparse.isspmatrix_bsr(A):
            A = A.tocsr()
        else:
            warn('implicit conversion to CSR', sparse.SparseEfficiencyWarning)
            A = sparse.csr_matrix(A)
    else:
        if sparse.isspmatrix(A) and A.format in formats:
            pass
        else:
            A = sparse.csr_matrix(A).asformat(formats[0])
  
    if not isinstance(x, np.ndarray):
        raise ValueError('expected numpy array for argument x')
    if not isinstance(b, np.ndarray):
        raise ValueError('expected numpy array for argument b')

    M, N = A.shape
  
    if M != N:
        raise ValueError('expected square matrix')

    if x.shape not in [(M,), (M, 1)]:
        raise ValueError('x has invalid dimensions')
    if b.shape not in [(M,), (M, 1)]:
        raise ValueError('b has invalid dimensions')

    if A.dtype != x.dtype or A.dtype != b.dtype:
        raise TypeError('arguments A, x, and b must have the same dtype')

    if not x.flags.carray:
        raise ValueError('x must be contiguous in memory')

    x = np.ravel(x)
    b = np.ravel(b)

    return A, x, b






def gauss_seidel(A, x, b, iterations=1, sweep='forward'):
    """Perform Gauss-Seidel iteration on the linear system Ax=b.

    Parameters
    ----------
    A : csr_matrix, bsr_matrix
        Sparse NxN matrix
    x : ndarray
        Approximate solution (length N)
    b : ndarray
        Right-hand side (length N)
    iterations : int
        Number of iterations to perform
    sweep : {'forward','backward','symmetric'}
        Direction of sweep

    Returns
    -------
    Nothing, x will be modified in place.

    Examples

    """
    A, x, b = make_system(A, x, b, formats=['csr', 'bsr'])

    if sparse.isspmatrix_csr(A):
        blocksize = 1
    else:
        R, C = A.blocksize
        if R != C:
            raise ValueError('BSR blocks must be square')
        blocksize = R

    if sweep == 'forward':
        row_start, row_stop, row_step = 0, int(len(x)/blocksize), 1
    elif sweep == 'backward':
        row_start, row_stop, row_step = int(len(x)/blocksize)-1, -1, -1
    elif sweep == 'symmetric':
        for iter in range(iterations):
            gauss_seidel(A, x, b, iterations=1, sweep='forward')
            gauss_seidel(A, x, b, iterations=1, sweep='backward')
        return
    else:
        raise ValueError("valid sweep directions are 'forward',\
                          'backward', and 'symmetric'")

    if sparse.isspmatrix_csr(A):
        for iter in range(iterations):
            pfas_core.gauss_seidel(A.indptr, A.indices, A.data, x, b,
                                  row_start, row_stop, row_step)
    else:
        for iter in range(iterations):
            pfas_core.bsr_gauss_seidel(A.indptr, A.indices, np.ravel(A.data),
                                      x, b, row_start, row_stop, row_step, R)

def projected_gauss_seidel(A, x, b, iterations=1, sweep='forward'):
  """Perform projected Gauss-Seidel iteration on the linear system Ax=b.

    Parameters
    ----------
    A : csr_matrix, bsr_matrix
        Sparse NxN matrix
    x : ndarray
        Approximate solution (length N)
    b : ndarray
        Right-hand side (length N)
    iterations : int
        Number of iterations to perform
    sweep : {'forward','backward','symmetric'}
        Direction of sweep

    Returns
    -------
    Nothing, x will be modified in place.

    Examples


    """
  A, x, b = make_system(A, x, b, formats=['csr', 'bsr'])

  if sparse.isspmatrix_csr(A):
      blocksize = 1
  else:
      R, C = A.blocksize
      if R != C:
          raise ValueError('BSR blocks must be square')
      blocksize = R

  if sweep == 'forward':
      row_start, row_stop, row_step = 0, int(len(x)/blocksize), 1
  elif sweep == 'backward':
      row_start, row_stop, row_step = int(len(x)/blocksize)-1, -1, -1
  elif sweep == 'symmetric':
      for iter in range(iterations):
          gauss_seidel(A, x, b, iterations=1, sweep='forward')
          gauss_seidel(A, x, b, iterations=1, sweep='backward')
      return
  else:
      raise ValueError("valid sweep directions are 'forward',\
                        'backward', and 'symmetric'")

  if sparse.isspmatrix_csr(A):
      for iter in range(iterations):
          pfas_core.projected_gauss_seidel(A.indptr, A.indices, A.data, x, b,
                                row_start, row_stop, row_step)
  else:
      raise TypeError('PGS must use csr matrix')



# from pyamg.utils import dispatcher
# dispatch = dispatcher( dict([ (fn,eval(fn)) for fn in __all__ ]) )
