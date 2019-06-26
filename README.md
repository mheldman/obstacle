Library of solvers for the discrete obstacle problem, which can be formulated as the following linear complementarity problem:

Find `u` in `R^N` such that

                        -Au >= f
                         u  >=  psi
          (Au + f)(u - psi) = 0,

where `A` is a discretization of the Laplacian on an `N`-point grid and `f` is an `N`-vector.
               
The solvers have a particular focus on multigrid methods, which are known to be efficient for solving elliptic PDEs like the Poisson equation. The multigrid-based projected full-approximation scheme method ([1]) and standard monotone multigrid method [2] implementations here are able to quickly solve some example problems with millions or ten of millions of unknowns on a desktop computer. 

Most of the library is written in Python using `scipy.sparse`, while the smoothers for the multigrid methods use extension modules written in `C++`. The code dealing with the setup, compilation, and installation of the extension modules is mostly borrowed from the library of algebraic multigrid solvers `pyamg` ([3]). 

To install, just run `sudo python setup.py install`. To test the installation, you can use the `test.py` file. The `test.py` file is configured to read input data for the obstacle problem from a `.py` file (stored in the same folder as `test.py`) specified at the commandline. For example, the following code would solve the dam problem, specified in `obstacle/obstacle/test/dam.py`, using an eight-grid V(1,1) cycle of the monotone multigrid method with coarse grid size (3 x 3):

`python test.py --solver_type monotone --problem_data dam --coarse_mx 1 --coarse_my 1 --show_residuals true --num_grids 8 --cycle_type V --smoothing_iters 1`

[1] Achi Brandt and Colin W. Cryer. Multigrid algorithms for the solution of linear complementarity problems
     arising from free boundary problems. Siam Journal on Scientific and Statistical Computing, 4(4):655–684, 1983.
     
[2] Ralf Kornhuber. Monotone multigrid methods for elliptic variational
inequalities I. Numerische Mathematik, 1994.
     
[3]  L. N. Olson and J.B. Schroder. PyAMG: Algebraic Multigrid Solvers in Python v4.0. https://github.com/pyamg/pyamg. 2018.
     

