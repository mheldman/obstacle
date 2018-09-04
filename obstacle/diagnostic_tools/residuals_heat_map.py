from diagnostics import radialproblem_pfas_diagnostic, radialproblem_rsp_diagnostic, damproblem_pfas_diagnostic, damproblem_rsp_diagnostic, unconstrproblem_pfas_diagnostic, unconstrproblem_multigrid_diagnostic
import numpy as np
import matplotlib.pyplot as plt
levels=9
mx, my = 1, 1
u, obstacle_p = radialproblem_pfas_diagnostic('V', coarse_mx = mx, coarse_my = my, min_num_levels = levels, max_num_levels = None, step = 1)
#u, obstacle_p = damproblem_pfas_diagnostic('V', coarse_mx = mx, coarse_my = my, min_num_levels = levels, max_num_levels = None, step = 1)

x1,x2,y1,y2 = 0.0,16.0,0.0,24.0
b = obstacle_p.F
A = obstacle_p.A
F = b - A.dot(u)
Fomega = np.minimum(F, 0.0)
Fomega[u > 0.0] = F[u > 0.0]
for i in range(levels-1):
    mx = 2*mx + 1
    my = 2*my + 1
z1 = np.linspace(x1,x2,mx + 2)
z2 = np.linspace(y1,y2,my + 2)
N = (mx+2)*(my + 2)
x, y  = np.zeros(N), np.zeros(N)
for i in range(my + 2):
    x[i*(mx + 2):(i+1)*(mx+2)] = z1
for j in range(my + 2):
    y[j*(mx+2):(j+1)*(mx+2)] = z2[j]
Fomega = abs(Fomega)
plt.ioff()
plt.close()
plt.scatter(x, y, c=abs(Fomega), cmap='inferno', linewidths=0.0, vmin = min(Fomega), vmax=max(Fomega))
plt.xlim(x1, x2)
plt.ylim(y1, y2)
plt.colorbar()
plt.show()

