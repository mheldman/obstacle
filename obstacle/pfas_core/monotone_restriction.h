#ifndef MONOTONE_RESTRICT_H
#define MONOTONE_RESTRICT_H

/*
 *  Perform monotone restriction on a two-dimensional obstacle
 
 *  Parameters  FIXME
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
template<class I, class F>
void monotone_restrict_2d(F ucoarse[], const int ucoarse_size,
                          const F ufine[], const int ufine_size,
                          const I mx, const I my,
                          const I mxcoarse, const I mycoarse)
{
  I index = 0;
  for(I j = 0; j != mycoarse + 2; j++) {
    for(I i = 0; i != mxcoarse + 2; i++) {
      if(i == mxcoarse + 1 || i == 0 || j == 0 || j == mycoarse + 1)
          ucoarse[index] = -1.0;
      else
      {
        ucoarse[index] = -1e20;
        I xindex;
        I yindex;
        I fine_index;
        for(I y = -1; y != 2; y++){
          for(I x = -1; x != 2; x++){
            xindex = 2*i + x;
            yindex = 2*j + y;
            fine_index = yindex*(mx + 2) + xindex;
            if(ufine[fine_index] > ucoarse[index])
              ucoarse[index] = ufine[fine_index];
            }
          }
      
      }
      index++;
        }
    
  }
  
}






#endif
