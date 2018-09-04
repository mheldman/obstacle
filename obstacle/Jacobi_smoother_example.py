from numpy import zeros, pi, dot, linspace, sin, ones
import matplotlib.pyplot as plt
import numpy as np
from GS import gs
import matplotlib

def build_poisson_1D(m):
    A = zeros((m, m))
    for i in range(1, m - 1):
        A[i, (i - 1, i, i + 1)] = 1.0, -2.0, 1.0
    A[0, (0, 1)] = -2.0, 1.0
    A[m - 1, (m - 2, m - 1)] = 1.0, -2.0
    F = zeros(m)
    F[0] = -1.0
    F[m - 1] = -1.0
    h = 1.0
    F = F/h**2
    A = A/h**2
    return A, F

A, F = build_poisson_1D(11)
X = linspace(0, 12, 13)
v1 = sin(X*pi/12)
v2 = sin(3*X*pi/12)
v3 = sin(11*X*pi/12)
v4 = (1/3)*(v1 - v2 + v3)

plt.plot(X, v1, 'r', label = '1')
plt.plot(X, v2, 'b',label = '2')
plt.plot(X, v3, 'g', label = '3')
plt.plot(X, v4, 'm', label = '4')
legend = plt.legend(loc='best', shadow=True, numpoints = 1)
plt.draw()
plt.show()

def add_boundary(v):
    x = zeros(len(v) + 2)
    for i in range(1,len(v) + 1):
        x[i] = v[i - 1]
    return x

def format_coords_for_latex(v, n, scale = 2, x = None):
    if x is None:
        x = np.linalg.norm(v)
    string = '{'
    for i in range(n):
        if i < n - 1:
            string += str(12*i/(n+1)) + '/' + str(scale*abs(v[i])**2) + ', '
        else:
            string += str(12*i/(n+1)) + '/' + str(scale*abs(v[i])**2)
    string += '}'
    return string

def format_lines_for_latex(v, scale = 250, options = 'blue', x = None):
    if x is None:
        x = np.linalg.norm(v)
    v = v / x
    string = ''
    for i in range(1,n):
        string += '\draw[' + options + '] (' + str(12*(i - 1)/(n+1)) + ', ' + str(scale*abs(v[i - 1])**2) + ')'
        string += ' -- '
        string += '(' + str(12*i/(n+1)) + ', ' + str(scale*abs(v[i])**2) + ');\n'
    return string

def format_lines_long(v, scale = 2, options = 'blue', n=100, x = None):
    if x is None:
        return format_lines_for_latex(v, options=options)
    else:
        return format_lines_for_latex(v, options=options, x=x)


normlist = [[], [], [], []]

v = [v1[1:12], v2[1:12], v3[1:12], v4[1:12]]
b = zeros(11)
u = []

for i in range(4):
    normlist[i].append(np.linalg.norm(v[i], np.inf))

for i in range(4):
    u.append(gs(A, v[i], b, maxiters=3))
    plt.plot(X, add_boundary(v[i]), label = 'vec' + str(i))
    normlist[i].append(np.linalg.norm(v[i], np.inf))
    #print(format_coords_for_latex(add_boundary(v[i])))
    #format_lines_for_latex(add_boundary(v[i]))

legend = plt.legend(loc='lower left', shadow=True, numpoints = 1)
plt.draw()
plt.show()

for i in range(4):
    u.append(gs(A, v[i], b, maxiters=12))
    plt.plot(X, add_boundary(v[i]), label = 'vec' + str(i))
    normlist[i].append(np.linalg.norm(v[i], np.inf))
    #print(format_coords_for_latex(add_boundary(v[i])))
    #format_lines_for_latex(add_boundary(v[i]), options = 'red')

legend = plt.legend(loc='lower left', shadow=True, numpoints = 1)
plt.draw()
plt.show()


for i in range(4):
    u.append(gs(A, v[i], b, maxiters=10))
    plt.plot(X, add_boundary(v[i]), label = 'vec' + str(i))
    normlist[i].append(np.linalg.norm(v[i], np.inf))
    #print(format_coords_for_latex(add_boundary(v[i])))
    #format_lines_for_latex(add_boundary(v[i]), options='green')

legend = plt.legend(loc='lower left', shadow=True, numpoints = 1)
plt.draw()
plt.show()

print(np.mean([normlist[0][1],normlist[2][1],normlist[1][1]]))
print(normlist[3][1])
n = 300
A, F = build_poisson_1D(n)

def build_ft_matrix(n):
    v = np.arange(n)+ 1
    A = np.zeros((n, n))
    for j in range(n):
        A[:, j] = np.sin(pi*(j + 1)*v/100)
    return A/np.sqrt(n/2)

X = build_ft_matrix(n)
#print(np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8)))
v = np.random.randn(n)
b = zeros(n)
mu = np.fft.fft(v)
#mu = np.linalg.solve(X, v)
gs(A, v, b, maxiters=1)
#mu2 = np.linalg.solve(X, v)
mu2 = np.fft.fft(v)
plt.plot(abs(mu[0:int(n/2)]))
plt.plot(abs(mu2[0:int(n/2)]))
gs(A, v, b, maxiters=2)

mu3 = np.fft.fft(v)
plt.plot(abs(mu3[0:int(n/2)]))
plt.show()
plt.plot(abs(mu[0:int(n/2)])/abs(mu[0:int(n/2)]))
plt.plot(abs(mu2[0:int(n/2)])/abs(mu[0:int(n/2)]))
plt.plot(abs(mu3[0:int(n/2)])/abs(mu[0:int(n/2)]))
plt.show()

text_file = open("tex_code_smoother_example.txt", "w")
text_file.write(format_lines_long(mu, n, options='black'))
text_file.close()

text_file = open("tex_code_smoother_example.txt", "a")
text_file.write(format_lines_long(mu2, n, options='blue',x=np.linalg.norm(mu)))
text_file.close()

text_file = open("tex_code_smoother_example.txt", "a")
text_file.write(format_lines_long(mu3, n, options='red',x=np.linalg.norm(mu)))
text_file.close()



