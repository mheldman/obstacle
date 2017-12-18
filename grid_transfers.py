from scipy import sparse

def interpolate(mx, my):

    mx_fine, my_fine = 2 * mx + 1, 2 * my + 1
    N, N_fine = (mx + 2) * (my + 2), (mx_fine + 2) * (my_fine + 2)
    P = sparse.lil_matrix((N_fine, N))
    k1, k2, k3, k4 = 0, 0, 0, 0

    for i in range(0, N_fine):
        if (i // (mx_fine + 2)) % 2 == 0:
            if (i - (i // (mx_fine + 2))) % 2 == 0:
                P[i, k1] = 4.0
                if i % (mx_fine + 2) == 0 and i > 0:
                    k2 = k1
                k1 += 1
            else:
                P[i, (k2, k2 + 1)] = 2.0, 2.0
                k2 += 1
        else:
            if (i - (i // (mx_fine + 2))) % 2 == 0:
                P[i, (k3, k3 + mx + 2)] = 2.0, 2.0
                if i % (mx_fine + 2) == 0 and i > 0:
                    k4 = k3
                k3 += 1
            else:
                P[i, (k4, k4 + 1, k4 + mx + 2, k4 + mx + 3)] = 1.0, 1.0, 1.0, 1.0
                k4 += 1

    P = P.tocsr()
    return P / 4.0

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
    N, N_coarse = (mx + 2) * (my + 2), (mx_coarse + 2) * (my_coarse + 2)
    R = sparse.lil_matrix((N_coarse, N))
    k = 0

    for i in range(N_coarse):
        if i % (mx_coarse + 2) == 0 and i > 0:
            k += 1
        n = 2 * i + k * (mx + 1)
        if (i - i * (mx_coarse + 2)) % 2 == 0:
            R[i, n] = 1.0

    R = R.tocsr()
    return R


