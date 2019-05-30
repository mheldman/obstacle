import numpy as np

x1, x2, y1, y2 = 0.0, 16.0, 0.0, 24.0
bounds = (x1, x2, y1, y2)
c = 4.0
f = lambda x, y: np.ones(len(x))
psi = lambda x, y: np.zeros(len(x))

def g(x, y):
    G = np.zeros(len(x))
    a = x2 - x1
    b = y2 - y1
    bc1 = (x == x1)
    G[bc1] = .5 * (b - y[bc1]) ** 2
    bc2 = (y <= c) & (x == x2)
    G[bc2] = .5 * (c - y[bc2]) ** 2
    bc3 = (y == y1)
    G[bc3] = ((a - x[bc3]) * b ** 2 + x[bc3] * c ** 2) / (2 * a)
    return G

uexact = None


