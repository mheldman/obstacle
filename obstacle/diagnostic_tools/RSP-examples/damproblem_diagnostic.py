from test_diagnostic import rsp_test
from numpy import linspace, zeros
import numpy as np
import matplotlib.pyplot as plt

x1 = 0.0
x2 = 16.0
y1 = 0.0
y2 = 24.0
c = 4.0

f = lambda x, y: 1.0
psi = lambda x, y: 0.0*x
def g(x, y):
    a = x2 - x1
    if x == x1:
        return .5 * (y2 - y)**2
    elif y <= c and x == x2:
        return .5 * (c - y)**2
    elif y == y1:
        return ((a - x) * y2**2 + x * c**2) / (2*a)
    else:
        return 0.0

bounds = (x1, x2, y1, y2)
mx = 500
my = 500
U = rsp_test(bounds, f, g, psi, mx, my, 'splu')

X = linspace(x1, x2, mx + 2)
Y = linspace(y1, y2, my + 2)
U1, U2, U3, U4, U5 = zeros(7), zeros(7), zeros(7), zeros(7), zeros(7)
k1, k2, k3, k4, k5 = 0, 0, 0, 0, 0
kk = lambda i, j: j * (mx + 2) + i
for j in range(my + 2):
    for i in range(mx + 2):
        k = kk(i, j)
        if X[i] == 0:
            if Y[j] in [n for n in range(0, 25, 4)]:
                U1[k1] = U[k]
                k1 += 1
        elif X[i] == 4:
            if Y[j] in [n for n in range(0, 25, 4)]:
                U2[k2] = U[k]
                k2 += 1
        elif X[i] == 8:
            if Y[j] in [n for n in range(0, 25, 4)]:
                U3[k3] = U[k]
                k3 += 1
        elif X[i] == 12:
            if Y[j] in [n for n in range(0, 25, 4)]:
                U4[k4] = U[k]
                k4 += 1
        elif X[i] == 16:
            if Y[j] in [n for n in range(0, 25, 4)]:
                U5[k5] = U[k]
                k5 += 1


U1exact = np.array([288, 200, 128, 72, 32, 8, 0])
U2exact = np.array([218, 146.5702, 89.9564, 47.2732, 18.1486, 2.5371, 0])
U3exact = np.array([148, 94.3247, 53.9823, 24.9879, 6.7841, 0, 0])
U4exact = np.array([78, 44.7462, 22.6601, 7.9120, 0, 0, 0])
U5exact = np.array([8, 0, 0, 0, 0, 0, 0])
exact = (U1exact, U2exact, U3exact, U4exact, U5exact)
Uexact = np.concatenate(exact)
num = [U1, U2, U3, U4, U5]
for i in range(5):
    print('Error on region ' + str(i + 1) + ':', np.linalg.norm(exact[i] - num[i], np.inf))
    print('Numerical solution on region ' + str(i + 1) + ':', num[i])
    print('Exact solution on region ' + str(i + 1) + ':', exact[i], '\n')

#print('Error on entire grid:', np.linalg.norm(Uexact - U, np.inf))


plt.close()
X = np.linspace(0.0,16.0,mx+2)
Y = np.linspace(0.0,24.0,my+2)
A, B = np.meshgrid(X, Y)
C = psi(A, B)

Z = np.zeros_like(C)
for i in range(my + 2):
    for j in range(mx + 2):
        Z[i, j] = U[kk(j, i)]
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.ioff()
fig = plt.figure()
ax = fig.gca(projection='3d')
surf1 = ax.plot_surface(A, B, C, color = 'b',vmin = -.5, vmax = 1.0, alpha = 1)
surf2 = ax.plot_surface(A, B, Z, color = 'r',vmin = -.5, vmax = 1.0, alpha = .35, cmap='viridis')
#ax.set_zlim(-.5, 1)
plt.pause(5)
plt.show()

