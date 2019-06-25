#ifndef RELAXATION_H
#define RELAXATION_H

/*
 *  Perform one iteration of Gauss-Seidel relaxation on the linear
 *  system Ax = b, where A is stored in CSR format and x and b
 *  are column vectors.
 *
 *  The unknowns are swept through according to the slice defined
 *  by row_start, row_end, and row_step.  These options are used
 *  to implement standard forward and backward sweeps, or sweeping
 *  only a subset of the unknowns.  A forward sweep is implemented
 *  with gauss_seidel(Ap, Aj, Ax, x, b, 0, N, 1) where N is the
 *  number of rows in matrix A.  Similarly, a backward sweep is
 *  implemented with gauss_seidel(Ap, Aj, Ax, x, b, N, -1, -1).
 *
 *  Parameters
 *      Ap[]       - CSR row pointer
 *      Aj[]       - CSR index array
 *      Ax[]       - CSR data array
 *      x[]        - approximate solution
 *      b[]        - right hand side
 *      row_start  - beginning of the sweep
 *      row_stop   - end of the sweep (i.e. one past the last unknown)
 *      row_step   - stride used during the sweep (may be negative)
 *
 *  Returns:
 *      Nothing, x will be modified in place
 *
 */
template<class I, class T, class F>
void gauss_seidel(const I Ap[], const int Ap_size,
                  const I Aj[], const int Aj_size,
                  const T Ax[], const int Ax_size,
                        T  x[], const int  x_size,
                  const T  b[], const int  b_size,
                  const I row_start,
                  const I row_stop,
                  const I row_step)
{
    for(I i = row_start; i != row_stop; i += row_step) {
        I start = Ap[i];
        I end   = Ap[i+1];
        T rsum = 0;
        T diag = 0;

        for(I jj = start; jj < end; jj++){
            I j = Aj[jj];
            if (i == j)
                diag  = Ax[jj];
            else
                rsum += Ax[jj]*x[j];
        }

        if (diag != (F) 0.0){
            x[i] = (b[i] - rsum)/diag;
        }
    }
}

/*
 *  Perform one iteration of projected Gauss-Seidel relaxation on the linear
 *  complementarity system b >= Ax, x >= 0, (b - Ax).T*x = 0, where A is stored in
 *  CSR format and x and b are column vectors.
 *
 *  The unknowns are swept through according to the slice defined
 *  by row_start, row_end, and row_step.  These options are used
 *  to implement standard forward and backward sweeps, or sweeping
 *  only a subset of the unknowns.  A forward sweep is implemented
 *  with gauss_seidel(Ap, Aj, Ax, x, b, 0, N, 1) where N is the
 *  number of rows in matrix A.  Similarly, a backward sweep is
 *  implemented with gauss_seidel(Ap, Aj, Ax, x, b, N, -1, -1).
 *
 *  Parameters
 *      Ap[]       - CSR row pointer
 *      Aj[]       - CSR index array
 *      Ax[]       - CSR data array
 *      x[]        - approximate solution
 *      b[]        - right hand side
 *      row_start  - beginning of the sweep
 *      row_stop   - end of the sweep (i.e. one past the last unknown)
 *      row_step   - stride used during the sweep (may be negative)
 *
 *  Returns:
 *      Nothing, x will be modified in place
 *
 */

template<class I, class T, class F>
void projected_gauss_seidel(const I Ap[], const int Ap_size,
                  const I Aj[], const int Aj_size,
                  const F Ax[], const int Ax_size,
                        F  x[], const int  x_size,
                  const F  b[], const int  b_size,
                  const F  p[], const int  p_size,
                  const I row_start,
                  const I row_stop,
                  const I row_step)
{
    for(I i = row_start; i != row_stop; i += row_step) {
        I start = Ap[i];
        I end   = Ap[i+1];
        F rsum = 0;
        F diag = 0;

        for(I jj = start; jj < end; jj++){
            I j = Aj[jj];
            if (i == j)
                diag  = Ax[jj];
            else
                rsum += Ax[jj]*x[j];
        }

        if (diag != (F) 0.0){
            x[i] = (b[i] - rsum)/diag;
            if(x[i] < p[i])
          {
              x[i] = p[i];
          }
        }
    }
}






#endif
