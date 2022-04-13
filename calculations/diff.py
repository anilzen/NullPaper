####################################
# Differentiation matrices
####################################

import scipy.sparse as sp

# Finite Difference
def Diff_mat_1D_o2(Nx, dx, periodic=False):

    # First derivative
    D_1d = sp.diags([-1, 1], [-1, 1], shape=(Nx, Nx))
    D_1d = sp.lil_matrix(D_1d)
    if periodic:
        D_1d[0, -1] = -1  #
        D_1d[-1, 0] = 1  #
    else:
        D_1d[0, [0, 1, 2]] = [-3, 4, -1]
        D_1d[-1, [-3, -2, -1]] = [1, -4, 3]

    # Second derivative
    D2_1d = sp.diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx))
    D2_1d = sp.lil_matrix(D2_1d)
    if periodic:
        D2_1d[0, -1] = 1
        D2_1d[-1, 0] = 1
    else:
        D2_1d[0, [0, 1, 2, 3]] = [2, -5, 4, -1]
        D2_1d[-1, [-4, -3, -2, -1]] = [-1, 4, -5, 2]

    return D_1d/(2*dx), D2_1d/(dx**2)


def Diff_mat_1D_o4(Nx, dx, periodic=False):

    # First derivative
    D_1d = sp.diags([1, -8, 8, -1], [-2, -1, 1, 2], shape=(Nx, Nx))
    D_1d = sp.lil_matrix(D_1d)
    if periodic:
        D_1d[0, [-1, -2]] = [-8, 1] 
        D_1d[1, [-1]] = [1]  
        D_1d[-1, [0, 1]] = [8, -1] 
        D_1d[-2, [0]] = [-1]  
    else:
        D_1d[0, [0, 1, 2, 3, 4]] = [-25, 48, -36, 16, -3]
        D_1d[1, [0, 1, 2, 3, 4]] = [-3, -10,  18, -6, 1]
        D_1d[-1, [-5, -4, -3, -2, -1]] = [3, -16, 36, -48, 25]
        D_1d[-2, [-5, -4, -3, -2, -1]] = [-1, 6, -18, 10, 3]

    # Second derivative
    D2_1d = sp.diags(
        [-1, 16, -30, 16, -1], [-2, -1, 0, 1, 2], shape=(Nx, Nx)
    )
    D2_1d = sp.lil_matrix(D2_1d)
    if periodic:
        D2_1d[0, [-1, -2]] = [16, -1]
        D2_1d[1, [-1]] = [-1]
        D2_1d[-1, [0, 1]] = [16, -1]
        D2_1d[-2, [0]] = [-1]
    else:
        D2_1d[0, [0, 1, 2, 3, 4, 5]] = [45, -154, 214, -156, 61, -10]
        D2_1d[1, [0, 1, 2, 3, 4, 5]] = [10, -15, -4, 14, -6, 1]
        D2_1d[-1, [-6, -5, -4, -3, -2, -1]] = [-10, 61, -156, 214, -154, 45]
        D2_1d[-2, [-6, -5, -4, -3, -2, -1]] = [1, -6, 14, -4, -15, 10]

    return D_1d/(12*dx), D2_1d/(12*dx**2)


def Diff_mat_1D_o6(Nx, dx, periodic=False):

    # First derivative
    D_1d = sp.diags(
        [-1, 9, -45, 45, -9, 1], [-3, -2, -1, 1, 2, 3], shape=(Nx, Nx)
    )
    D_1d = sp.lil_matrix(D_1d)
    if periodic:
        D_1d[0, [-1, -2, -3]] = [-45, 9, -1]  #
        D_1d[1, [-1, -2]] = [9, -1]  #
        D_1d[2, [-1]] = [-1]  #
        D_1d[-1, [0, 1, 2]] = [45, -9, 1]  #
        D_1d[-2, [0, 1]] = [-9, 1]  #
        D_1d[-3, [0]] = [1]  #
    else:
        D_1d[0, [0, 1, 2, 3, 4, 5, 6]] = [-147, 360, -450, 400, -225, 72, -10]
        D_1d[1, [0, 1, 2, 3, 4, 5, 6]] = [-10, -77, 150, -100, 50, -15, 2]
        D_1d[2, [0, 1, 2, 3, 4, 5, 6]] = [2, -24, -35, 80, -30, 8, -1]
        D_1d[-1, [-7, -6, -5, -4, -3, -2, -1]] = [10, -72, 225, -400, 450, -360, 147]
        D_1d[-2, [-7, -6, -5, -4, -3, -2, -1]] = [-2, 15, -50, 100, -150, 77, 10]
        D_1d[-3, [-7, -6, -5, -4, -3, -2, -1]] = [1, -8, 30, -80, 35, 24, -2]

    # Second derivative
    D2_1d = sp.diags(
        [2, -27, 270, -490, 270, -27, 2], [-3, -2, -1, 0, 1, 2, 3], shape=(Nx, Nx)
    )
    D2_1d = sp.lil_matrix(D2_1d)
    if periodic:
        D2_1d[0, [-1, -2, -3]] = [270, -27, 2]  #
        D2_1d[1, [-1, -2]] = [-27, 2]  #
        D2_1d[2, [-1]] = [2]  #
        D2_1d[-1, [0, 1, 2]] = [270, -27, 2]  #
        D2_1d[-2, [0, 1]] = [-27, 2]  #
        D2_1d[-3, [0]] = [2]  #
    else:
        D2_1d[0, [0, 1, 2, 3, 4, 5, 6, 7]] = [938, -4014, 7911, -9490, 7380, -3618, 1019, -126]
        D2_1d[1, [0, 1, 2, 3, 4, 5, 6, 7]] = [126, -70, -486, 855, -670, 324, -90, 11]
        D2_1d[2, [0, 1, 2, 3, 4, 5, 6, 7]] = [-11, 214, -378, 130, 85, -54, 16, -2]
        D2_1d[-1, [-8, -7, -6, -5, -4, -3, -2, -1]] = [-126, 1019, -3618, 7380, -9490, 7911, -4014, 938]
        D2_1d[-2, [-8, -7, -6, -5, -4, -3, -2, -1]] = [11, -90, 324, -670, 855, -486, -70, 126]
        D2_1d[-3, [-8, -7, -6, -5, -4, -3, -2, -1]] = [-2, 16, -54, 85, 130, -378, 214, -11]
    return D_1d/(60*dx), D2_1d/(180*dx**2)


# def Diff_mat_2D(Nx, Ny, y_periodic=False, y_order=2):

#     # 1D differentiation matrices
#     Dx_1d, D2x_1d = Diff_mat_1D(Nx)
#     if y_order == 2:
#         Dy_1d, D2y_1d = Diff_mat_1D(Ny, y_periodic)
#     elif y_order == 4:
#         Dy_1d, D2y_1d = Diff_mat_1D_o4(Ny, y_periodic)
#     elif y_order == 6:
#         Dy_1d, D2y_1d = Diff_mat_1D_o6(Ny, y_periodic)
#     else:
#         print("y_order " + str(y_order) + " has not been implemented")

#     # Sparse identity matrices
#     Ix = sp.eye(Nx)
#     Iy = sp.eye(Ny)

#     # 2D matrix operators from 1D operators using kronecker product
#     # First partial derivatives
#     Dx_2d = sp.kron(Iy, Dx_1d)
#     Dy_2d = sp.kron(Dy_1d, Ix)

#     # Second partial derivatives
#     D2x_2d = sp.kron(Iy, D2x_1d)
#     D2y_2d = sp.kron(D2y_1d, Ix)

#     # Return compressed Sparse Row format of the sparse matrices
#     return Dx_2d.tocsr(), Dy_2d.tocsr(), D2x_2d.tocsr(), D2y_2d.tocsr()


# def set_Diff_matrices(M, dr, dt, y_periodic=False, y_order=2):
#     # Construct the differentiation matrices
#     Dr, Dt, Dr2, Dt2 = Diff_mat_2D(M, M, y_periodic=y_periodic, y_order=y_order)
#     I_sp = sp.eye(M * M).tocsr()

#     Dr = Dr / (2 * dr)
#     Dr2 = Dr2 / dr ** 2

#     if y_order == 2:
#         Dt = Dt / (2 * dt)
#         Dt2 = Dt2 / dt ** 2
#     elif y_order == 4:
#         Dt = Dt / (12 * dt)
#         Dt2 = Dt2 / (12 * dt ** 2)
#     elif y_order == 6:
#         Dt = Dt / (60 * dt)
#         Dt2 = Dt2 / (180 * dt ** 2)
#     else:
#         print("y_order: " + str(y_order) + " not implemented")
#     return Dr, Dr2, Dt, Dt2


# # Collocation methods

# def cheb(N):
#     x = np.cos(np.pi * np.arange(0, N + 1) / N)
#     if N % 2 == 0:
#         x[N // 2] = 0.0
#     c = np.ones(N + 1)
#     c[0] = 2.0
#     c[N] = 2.0
#     c = c * (-1.0) ** np.arange(0, N + 1)
#     c = c.reshape(N + 1, 1)
#     X = np.tile(x.reshape(N + 1, 1), (1, N + 1))
#     dX = X - X.T
#     D = np.dot(c, 1.0 / c.T) / (dX + np.eye(N + 1))
#     D = D - np.diag(D.sum(axis=1))
#     return D, x


# def chebfft(v):
#     N = len(v) - 1
#     if N == 0:
#         w = 0.0  # only when N is even!
#         return w
#     x = cos(pi * arange(0, N + 1) / N)
#     ii = arange(0, N)
#     V = flipud(v[1:N])
#     V = list(v) + list(V)
#     U = real(fft(V))
#     b = list(ii)
#     b.append(0)
#     b = b + list(arange(1 - N, 0))
#     w_hat = 1j * array(b)
#     w_hat = w_hat * U
#     W = real(ifft(w_hat))
#     w = zeros(N + 1)
#     w[1:N] = -W[1:N] / sqrt(1 - x[1:N] ** 2)
#     w[0] = sum(ii ** 2 * U[ii]) / N + 0.5 * N * U[N]
#     w[N] = (
#         sum((-1) ** (ii + 1) * ii ** 2 * U[ii]) / N + 0.5 * (-1) ** (N + 1) * N * U[N]
#     )
#     return w
