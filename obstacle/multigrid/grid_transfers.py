from scipy import sparse
import numpy as np
import obstacle.pfas_core as pfas_core

def interpolate(mx, my):

    mx_fine, my_fine  = 2 * mx + 1, 2 * my + 1
    N, N_fine = (mx + 2) * (my + 2), (mx_fine + 2) * (my_fine + 2)
    P1 = sparse.lil_matrix((2*(mx_fine + 2), 2*(mx + 2)))

    P1[range(0, mx_fine + 2, 2), range(mx + 2)] = 4.0
    P1[range(1, mx_fine + 2, 2), range(0, mx + 1)] = 2.0
    P1[range(1, mx_fine + 2, 2), range(1, mx + 2)] = 2.0

    P1[range(mx_fine + 2, 2*(mx_fine + 2), 2), range(mx + 2)] = 2.0
    P1[range(mx_fine + 2, 2 * (mx_fine + 2), 2), range(mx + 2, 2 * (mx + 2))] = 2.0

    P1[range(mx_fine + 3, 2 * (mx_fine + 2), 2), range(mx + 1)] = 1.0
    P1[range(mx_fine + 3, 2 * (mx_fine + 2), 2), range(1, mx + 2)] = 1.0
    P1[range(mx_fine + 3, 2 * (mx_fine + 2), 2), range(mx + 2, 2*(mx + 1) + 1)] = 1.0
    P1[range(mx_fine + 3, 2 * (mx_fine + 2), 2), range(mx + 3, 2 * (mx + 1) + 2)] = 1.0

    P2 = P1[0:mx_fine + 2, 0:mx + 2]

    P2 = P2.tocsr()
    P2 /= 4.0
    P1 = P1.tocsr()
    P1 /= 4.0

    block_list = [sparse.csr_matrix((P1.data, P1.indices + i*(mx + 2), P1.indptr), shape=(P1.shape[0], N)) for i in range(my + 1)]
    block_list.append(sparse.csr_matrix((P2.data, P2.indices + (my + 1)*(mx + 2), P2.indptr), shape=(P2.shape[0], N)))
    P = sparse.vstack(block_list, format='csr')
    return P

def restrict_fw(mx, my):

    mx_coarse = (mx - 1) // 2
    my_coarse = (my - 1) // 2
    N = (mx + 2) * (my + 2)
    N_coarse = (mx_coarse + 2) * (my_coarse + 2)
    R = sparse.lil_matrix((N_coarse, N))
    k = 0

    for i in range(0, N_coarse):
        if i % (mx_coarse + 2) == 0 and i > 0:
            k += 1
        n = 2 * i + k * (mx + 1)
        if i < mx_coarse + 2:
            R[i, n] = 1.0
        if i > (mx_coarse + 1) * (my_coarse + 2):
            R[i, n] = 1.0
        if i % (mx_coarse + 2) == 0:
            R[i, n] = 1.0
        if i % (mx_coarse + 2) == mx_coarse + 1:
            R[i, n] = 1.0
        elif i > mx_coarse + 2 and i < (mx_coarse + 1) * (my_coarse + 2):
            R[i, (n - 1, n, n + 1)] = .125, .25, .125
            R[i, (n - mx - 2, n + mx + 2)] = .125, .125
            R[i, (n - mx - 3, n + mx + 1)] = .0625, .0625
            R[i, (n - mx - 1, n + mx + 3)] = .0625, .0625

    R = R.tocsr()
    return R


def restrict_inj(mx, my):
    mx_coarse, my_coarse = (mx - 1) // 2, (my - 1) // 2
    N = (mx + 2) * (my + 2)
    P1 = sparse.lil_matrix((mx_coarse + 2, mx + 2))
    P1[range(P1.shape[0]), range(0, mx + 2, 2)] = 1.0
    P1 = P1.tocsr()
    block_list = [sparse.csr_matrix((P1.data, P1.indices + 2*i*(mx + 2), P1.indptr), shape=(P1.shape[0], N)) for i in range(my_coarse + 2)]
    return sparse.vstack(block_list, format='csr')

def monotone_restrict2d(mx, my, u, u_coarse):
  mx_coarse = (mx - 1) // 2
  my_coarse = (my - 1) // 2
  pfas_core.monotone_restrict_2d(u_coarse, u, mx, my, mx_coarse, my_coarse)
  return u_coarse
