Library of solvers for the discrete obstacle problem, which can be formulated as the following linear complementarity problem:

Find u in R^N such that

                        -Au >= f
                         u  >=  psi
          (Au + f)(u - psi) = 0,

where A is a discretization of the Laplacian on an N-point grid and f is an N-vector.
               

The solvers have a particular focus on multigrid methods, which are known to be efficient for solving elliptic PDEs like the Poisson equation. While not yet parallel, the multigrid-based projected full-approximation scheme method ([1]) implementation here is able to quickly solve some example problems with millions or ten of millions of unknowns on a desktop computer. 

Most of the library is written in Python using `scipy.sparse`, while the smoothers for the multigrid methods use extension modules written in `C++`. The code dealing with the setup, compilation, and installation of the extension modules is mostly borrowed from the library of algebraic multigrid solvers `pyamg` ([2]). 

To install, just run `sudo python setup.py install`.

Note that there is a problem with the extension module installation which requires the user to add the location of the extension pfas_core to the Python path. This can be done by writing `export PYTHONPATH=.`, where `.` is the path to your pfas_core extension in your installation of obstacle. For example, if your obstacle package was installed using anaconda3 your path might look like `~/anaconda3/lib/python2.7/site-packages/(egg name)/obstacle/pfas_core`.

The `scratch_12.py` file is meant as a test for some of the solvers. It can run several examples for the PFAS method, the reduced space method, and the projected Newton method with different fine grid sizes. The best way to run the scratch file is to copy the file from your installation into a working directory (so that you can make changes) and run it from there. The examples themselves are contained in the file `diagnostics.py`. 

[1] Achi Brandt and Colin W. Cryer. Multigrid algorithms for the solution of linear complementarity problems
     arising from free boundary problems. Siam Journal on Scientific and Statistical Computing, 4(4):655–684, 1983.
     
[2]  L. N. Olson and J.B. Schroder. PyAMG: Algebraic Multigrid Solvers in Python v4.0. https://github.com/pyamg/pyamg. 2018.
     

