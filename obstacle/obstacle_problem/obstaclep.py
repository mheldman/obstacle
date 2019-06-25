import numpy as np
from numpy import zeros
from .Poisson2D import poisson2d, rhs

__all__ = ['box_obstacle_problem']

class box_obstacle_problem:

  def __init__(self, bounds, f, g, psi, A=None, U=None, P=None, F=None, mx=None, my=None):

      self.bounds = bounds
      self.f = f
      self.g = g
      self.psi = psi
      self.A = A
      self.U = U
      self.P = P
      self.F = F
      self.mx = mx
      self.my = my
      self.bndry_pts = []
      self.solution = None

  def __repr__(self):
      x1, x2, y1, y2 = self.bounds
      output = 'Obstacle problem on [' + str(x1) + ', ' + str(x2) + '] x [' + str(y1) + ', ' + str(y2) + ']\n'
      if self.mx is None:
          return output
      mx, my = self.mx, self.my
      output += 'Discretized on (' + str(mx + 2) + ' x ' + str(my + 2) + ') grid\n'
      return output

  def initialize(self):

      mx, my = self.mx, self.my
      x1, x2, y1, y2 = self.bounds
      X, Y = np.linspace(x1, x2, mx + 2), np.linspace(y1, y2, my + 2)
      N = (mx + 2) * (my + 2)
      psi = self.psi
      if type(psi(X, Y)) in [int, float]:
          U, P = zeros(N), zeros(N)
          kk = lambda i, j: j * (mx + 2) + i
          for j in range(my + 2):
              for i in range(mx + 2):
                  k = kk(i, j)
                  P[k] = psi(X[i], Y[j])
                  U[k] = max(-P[k], 0.0)
      else:
          [X, Y] = np.meshgrid(X, Y)
          P = psi(X.flatten(), Y.flatten())
          U = np.maximum(-P, zeros(N))

      self.F = self.F - self.A.dot(P)
      U[self.bndry_pts] = self.F[self.bndry_pts]
      self.U, self.P = U, P

  def discretize(self, mx, my):

      self.mx, self.my = mx, my
      (x1, x2, y1, y2) = self.bounds
      self.F = rhs(self.f, mx, my, g=self.g, x1=x1, x2=x2, y1=y1, y2=y2)
      self.A = poisson2d(mx, my, x1=x1, x2=x2, y1=y1, y2=y2)
      X, Y = np.arange(0, mx + 2), np.arange(0, my + 2)
      X, Y = np.meshgrid(X, Y)
      X, Y = X.flatten(), Y.flatten()
      bndry_vals = (X == 0) | (Y == my + 1) | (Y == 0) | (X == mx + 1)
      self.bndry_pts = np.arange(len(X))[bndry_vals]
      self.initialize()

  def solve(self, obstacle_solver, *args, **kwargs):
      print(self)
      self.U = obstacle_solver(*args, **kwargs)
      self.solution = self.U + self.P
      return self.solution

  def plot_solution(self):
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    mx, my = self.mx, self.my
    (x1, x2, y1, y2) = self.bounds
    X, Y = np.arange(0, mx + 2), np.arange(0, my + 2)
    [X, Y] = np.meshgrid(X, Y)
    Z = self.solution.reshape((my + 2, mx + 2))
    P = self.P.reshape((my + 2, mx + 2))
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z)
    ax.plot_surface(X, Y, P)
    plt.show()

  def plot_obstacle(self):

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    mx, my = self.mx, self.my
    (x1, x2, y1, y2) = self.bounds
    X, Y = np.arange(0, mx + 2), np.arange(0, my + 2)
    X, Y = np.meshgrid(X, Y)
    Z = self.solution.reshape((my + 2, mx + 2))
    P = self.P.reshape((my + 2, mx + 2))
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, P)
    plt.show()

  def plot_active_set(self):
  
    import matplotlib.pyplot as plt

    x1, x2, y1, y2 = self.bounds
    mx, my = self.mx, self.my
    Z = self.U.reshape((my + 2, mx + 2))
    X = np.linspace(x1, x2, mx + 2)
    Y = np.linspace(y1, y2, my + 2)
    A, B = np.meshgrid(X, Y)
    plt.plot(A[[Z < 1e-15]], B[[Z < 1e-15]], 'o', color='k')
    plt.xlim(x1, x2)
    plt.ylim(y1, y2)
    plt.show()
    plt.pause(2)
    plt.close('all')
