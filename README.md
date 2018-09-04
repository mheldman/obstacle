Library of obstacle problem solvers including multigrid-based methods. Most of the library is written in Python using scipy.sparse, while the smoothers for the multigrid methods run in C++. While not parallel, the multigrid-based projected full-approximation scheme method ([1]) is able to handle some example problems with over ten million unknowns in only a couple minutes. 

To install, just run `sudo python setup.py install`

Note that there is a problem with the extension module installation which requires the user to add the location of the extension pfas_core to the Python path. This can be done by writing `export PYTHONPATH=.`, where `.` is the path to your pfas_core extension in your installation of obstacle. For example, if your obstacle package was installed using anaconda3 your path might look like `~/anaconda3/lib/python2.7/site-packages/(egg name)/obstacle/pfas_core`.

The scratch_12.py file is meant as a test for some of the solvers. It can run several examples for the PFAS method, the reduced space method, and the projected Newton method with different fine grid sizes. The examples themselves are contained in the file diagnostics.py. 

[Achi Brandt and Colin W. Cryer. Multigrid algorithms for the solution of linear complementarity problems
     arising from free boundary problems. Siam Journal on Scientific and Statistical Computing, 4(4):655–684, 1983.][1]
     

