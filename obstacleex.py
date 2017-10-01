from obstacle import*
import numpy as np
import matplotlib.pyplot as plt
from blockprint import*
from time import time
from mpl_toolkits.mplot3d import Axes3D

m = 100
Alpha = .68026
Beta = .47152
psi = lambda x,y: np.sqrt(np.maximum(0.0, 1 - x**2 - y**2)) + np.minimum(0.0,1-x**2-y**2)
f = lambda x,y: x*0
g = lambda x,y: -Alpha*np.log(np.sqrt(x**2 + y**2)) + Beta

blockPrint()
start = time()
U1 = obstacleqp(psi, m, f, g, a = -2.0, b = 2.0)
time1 = time() - start
start = time()
U2 = obstaclersp(psi, m, f, g, a = -2.0, b = 2.0)
time2 = time() - start
start = time()
U3 = obstaclecpgs(psi, m, f, g, a = -2.0, b = 2.0)
time3 = time() - start
enablePrint()

Z1 = np.zeros((m,m))
Z2 = np.zeros((m,m))
Z3 = np.zeros((m,m))
Zexact = np.zeros((m,m))

h = 4/(m+1)
X = np.linspace(-2.0 + h, 2.0 - h, m)
N = m**2
Uexact = np.zeros((N,1))
kk = lambda i, j: j * m + i
for j in range(m):
    for i in range(m):
        r = np.sqrt(X[i]**2 + X[j]**2)
        k = kk(i, j)
        if r > .69797:  #if (X[i], X[j]) is outside the contact region
            Uexact[k] = g(X[i], X[j])
        else:
            Uexact[k] = psi(X[i], X[j])
        Z1[i, j] = U1[k]
        Z2[i, j] = U2[k]
        Z3[i, j] = U3[k]

err1 = np.linalg.norm(Uexact - U1, np.inf)
err2 = np.linalg.norm(Uexact - U2, np.inf)
err3 = np.linalg.norm(Uexact - U3, np.inf)

[A, B] = np.meshgrid(X, X)
P = psi(A, B)

print('Error for QP method is:', err1)
print('Time for QP method is', time1)
print('Error for RSP method is:', err2)
print('Time for RSP method is', time2)
print('Error for CPGS method is:', err3)
print('Time for CPGS method is', time3)

fig1 = plt.figure()
ax1 = fig1.gca(projection = '3d')
surf1 = ax1.plot_surface(A, B, P, color = 'r',vmin = 0.0, vmax = 5.1, alpha = .5)
surf2 = ax1.plot_surface(A,B,Z1,color = 'g',vmin = 0.0, vmax = 5.1, alpha = .5)
ax1.set_zlim3d(0,1)
ax1.set_xlim3d(-2,2)
ax1.set_ylim3d(-2,2)

fig2 = plt.figure()
ax2 = fig2.gca(projection = '3d')
surf3 = ax2.plot_surface(A, B, P, color = 'r',vmin = 0.0, vmax = 5.1, alpha = .5)
surf4 = ax2.plot_surface(A, B, Z2,color = 'g',vmin = 0.0, vmax = 5.1, alpha = .5)
ax2.set_zlim3d(0,1)
ax2.set_xlim3d(-2,2)
ax2.set_ylim3d(-2,2)

fig3 = plt.figure()
ax3 = fig3.gca(projection = '3d')
surf5 = ax3.plot_surface(A, B, P, color = 'r',vmin = 0.0, vmax = 5.1, alpha = .5)
surf6 = ax3.plot_surface(A, B, Z3,color = 'g',vmin = 0.0, vmax = 5.1, alpha = .5)
ax3.set_zlim3d(0,1)
ax3.set_xlim3d(-2,2)
ax3.set_ylim3d(-2,2)