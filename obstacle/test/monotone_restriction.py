import numpy as np
from obstacle.multigrid.grid_transfers import monotone_restrict2d
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

mx = 5
my = mx
u = np.random.randint(100,size=(mx+2)*(my+2))
u = u/1.0

mxcoarse = (mx - 1) // 2
mycoarse = (my - 1) // 2
u_coarse = np.zeros((mxcoarse + 2)*(mycoarse + 2))
monotone_restrict2d(mx, my, u, u_coarse)

fig = plt.figure()
ax = fig.add_subplot(111)

X = np.linspace(0,1,mx+2)
Y = X
u = u.reshape(mx + 2, my + 2)

for j in range(my + 2):
  for i in range(mx + 2):
    plt.plot(X[i], Y[j], 'ko')
    ax.annotate('%s' % u[j][i], xy=(X[i],Y[j]), textcoords='offset pixels')
plt.grid()
plt.savefig('/Users/maxheldman/obstacle/obstacle/figures/fine_grid.png')
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111)

mx = (mx - 1) // 2
my = (my - 1) // 2

u = u_coarse.reshape(mx + 2, my + 2)

for j in range(my + 2):
  for i in range(mx + 2):
    plt.plot(X[2*i], Y[2*j], 'ko')
    ax.annotate('%s' % u[j][i], xy=(X[2*i],Y[2*j]), textcoords='offset pixels')
plt.grid()
plt.savefig('/Users/maxheldman/obstacle/obstacle/figures/coarse_grid.png')

