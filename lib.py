import numpy as np
import tensorly as tl
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
import numpy as np
import matplotlib.pyplot as plt
from pydmd import HODMD
from pydmd.plotter import plot_eigs, plot_summary
import warnings
from scipy import io
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from pydmd import DMD, BOPDMD
from pydmd.plotter import plot_eigs, plot_summary
from pydmd.preprocessing import hankel_preprocessing
# from lib import *
from pydmd.preprocessing import hankel_preprocessing
import pydmd


def dynamic_mode_decomposition(xi, t, f, r=20,indices=[170, 199, 210]):
    """
    Perform Dynamic Mode Decomposition (DMD) on spatio-temporal data.

    Parameters:
        xi (np.ndarray): Spatial grid points.
        t (np.ndarray): Time points.
        f (np.ndarray): Spatio-temporal data matrix of shape (len(t), len(xi)).
        r (int): Rank for SVD truncation.
    """
    
    N_train = f.shape[0] // 2
    f_train = f[:N_train]
    t_train = t[:N_train]

    # Step 1: Build linear system
    X = f_train.T
    X1 = X[:, :-1]
    X2 = X[:, 1:]

    # Step 2: Singular Value Decomposition
    U, Sdiag, Vh = np.linalg.svd(X1, full_matrices=False)
    S = np.diag(Sdiag)
    V = np.conj(Vh).T

    Ur = U[:, :r]
    Sr = S[:r, :r]
    Vr = V[:, :r]
    # Step 3: Compute Atilde
    Atilde = np.dot(np.conj(Ur.T), np.dot(X2, np.dot(Vr, np.linalg.inv(Sr))))
    Lambda_diag, W = np.linalg.eig(Atilde)
    Phi = np.dot(X2, np.dot(Vr, np.dot(linalg.inv(Sr), W)))

    omega = np.log(Lambda_diag) / (t[1] - t[0])

    # Step 4: Reconstruct signal
    x1 = X[:, 0]
    b = np.linalg.pinv(Phi) @ x1

    t_dyn = np.zeros((r, len(t_train)), dtype=complex)
    for i in range(len(t_train)):
        t_dyn[:, i] = b * np.exp(omega * t_train[i])
    f_dmd = Phi @ t_dyn

    # Step 5: Predict future dynamics
    t_ext = t
    t_ext_dyn = np.zeros((r, len(t_ext)), dtype=complex)
    for i in range(len(t_ext)):
        t_ext_dyn[:, i] = b * np.exp(omega * t_ext[i])
    f_dmd_ext = Phi @ t_ext_dyn
    dmd = pydmd.DMD(svd_rank=r)
    dmd.fit(f)

    plot_singular_values(Sdiag)
    plot_dynamic_modes(xi, Phi)
    plot_eigenvalues(Lambda_diag)
    plot_time_modes(t_train, t_dyn)
    plot_original_vs_reconstructed_1d(t_train, xi, f_train, f_dmd, indices=indices)
    plot_2d_comparison(t_train, xi, f_train, f_dmd)
    plot_prediction(t, t_train, xi, f, f_dmd_ext, N_train, indices=indices)
    plot_2d_prediction(t_ext, xi, f_dmd_ext)
    return dmd

# Helper Plotting Functions

def plot_singular_values(S):
    plt.figure(figsize=(6, 6))
    plt.plot(S, ".")

    plt.yscale("log")
    plt.title("Singular Values (log scale)")
    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.grid(True)
    plt.show()


def plot_dynamic_modes(xi, Phi):
    plt.figure(figsize=(16, 6))
    plt.title("Dynamic Modes")
    for i in range(min(3, Phi.shape[1])):
        plt.plot(xi, np.real(Phi[:, i]), label=f'$\Phi_{i}$')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Mode amplitude")
    plt.grid(True)
    plt.show()


def plot_eigenvalues(Lambda_diag):
    theta = np.linspace(0, 2 * np.pi, 150)
    radius = 1
    a = radius * np.cos(theta)
    b = radius * np.sin(theta)

    plt.figure(figsize=(6, 6))
    for i in range(min(3, len(Lambda_diag))):
        plt.scatter(np.real(Lambda_diag[i]), np.imag(Lambda_diag[i]), label=f"$\lambda_{i}$")
    plt.plot(a, b, color='k', ls='--', label="Unit circle")
    plt.xlabel("$\lambda_r$")
    plt.ylabel("$\lambda_i$")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.title("Eigenvalues of $\\tilde{A}$")
    plt.show()


def plot_time_modes(t_train, t_dyn):
    plt.figure(figsize=(16, 6))
    plt.title("Time Modes")
    for i in range(min(3, t_dyn.shape[0])):
        plt.plot(t_train, np.real(t_dyn[i, :]), label=f'Time mode {i}')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


def plot_original_vs_reconstructed_1d(t_train, xi, f_train, f_dmd, indices):
    for idx in indices:
        plt.figure(figsize=(16, 6))
        plt.title(f"1D Plot at x = {idx}")
        plt.plot(t_train, np.real(f_train[:, idx]), label='Original data')
        plt.plot(t_train, np.real(f_dmd[idx, :]), '--', label='Reconstructed by DMD')
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("f(x,t)")
        plt.grid(True)
        plt.show()


def plot_2d_comparison(t_train, xi, f_train, f_dmd):
    plt.figure(figsize=(16, 6))
    plt.title('2D Plot: Original Data')
    plt.pcolormesh(t_train, xi, np.real(f_train.T), shading='auto')
    plt.colorbar(label='f(x,t)')
    plt.xlabel("Time")
    plt.ylabel("Space (x)")
    plt.show()

    plt.figure(figsize=(16, 6))
    plt.title('2D Plot: DMD Reconstruction')
    plt.pcolormesh(t_train, xi, np.real(f_dmd), shading='auto')
    plt.colorbar(label='f(x,t)')
    plt.xlabel("Time")
    plt.ylabel("Space (x)")
    plt.show()


def plot_prediction(t, t_train, xi, f, f_dmd_ext, N_train, indices):
    for idx in indices:
        plt.figure(figsize=(16, 6))
        plt.title(f"Prediction at x = {idx}")
        plt.plot(t, np.real(f[:, idx]), label='Original data')
        plt.plot(t_train, np.real(f_dmd_ext[idx, :N_train]), '--', label='Reconstructed by DMD')
        plt.plot(t[N_train:], np.real(f_dmd_ext[idx, N_train:]), '.', label='Predicted by DMD')
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("f(x,t)")
        plt.grid(True)
        plt.show()


def plot_2d_prediction(t_ext, xi, f_dmd_ext):
    plt.figure(figsize=(16, 6))
    plt.title('2D Plot: Extended Prediction')
    plt.pcolormesh(t_ext, xi, np.real(f_dmd_ext), shading='auto')
    plt.colorbar(label='f(x,t)')
    plt.xlabel("Time")
    plt.ylabel("Space (x)")
    plt.show()


def delayed_matrix(X, delay,vstack=True):
    n, t = X.shape
    cols = t - delay + 1
    X_delayed = [X[:, i:i+cols] for i in range(delay)]
    if not vstack:
        return np.hstack(X_delayed)
    return np.vstack(X_delayed)

def tolerance(S, threshold=1e-6):
    S_squared = S**2
    total_energy = np.sum(S_squared)
    cumulative = np.cumsum(S_squared[::-1])[::-1]
    EE = cumulative / total_energy
    N = np.argmax(EE <= threshold)
    return int(N)

def HODMD(X, delay, energy_threshold, dt):
    X_aug = delayed_matrix(X, delay)
    X1 = X_aug[:, :-1]
    X2 = X_aug[:, 1:]

    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    r = tolerance(S, energy_threshold)

    Ur = U[:, :r].astype(np.complex128)
    Sr = np.diag(S[:r]).astype(np.complex128)
    Vr = Vh.conj().T[:, :r].astype(np.complex128)

    Atilde = Ur.conj().T @ X2 @ Vr @ np.linalg.inv(Sr)
    Lambda, W = np.linalg.eig(Atilde)
    Phi = X2 @ Vr @ np.linalg.inv(Sr) @ W
    omega = np.log(Lambda) / dt

    alpha1 = np.linalg.lstsq(Phi, X1[:, 0], rcond=None)[0]

    time_dynamics = np.zeros((r, X1.shape[1]), dtype=np.complex128)
    print("x1shape",X1.shape[1])
    for i in range(X1.shape[1]):
        time_dynamics[:, i] = alpha1 * np.exp(omega * ((i + 1) * dt))

    X_dmd = Phi @ time_dynamics
    return X_dmd, r

def reconstruct_from_tt(tt_cores):
    """
    Reconstructs a full tensor from its TT decomposition.

    Args:
        tt_cores (list of np.ndarray): List of TT cores. Each core has shape (r_prev, d_mode, r_next).

    Returns:
        np.ndarray: Reconstructed full tensor.
    """
    tensor = tt_cores[0]  # Start with first core
    for i in range(1, len(tt_cores)):
        # Contract along the bond dimension
        tensor = np.tensordot(tensor, tt_cores[i], axes=([-1], [0]))

    # Get original dimensions from the cores
    dimensions = [core.shape[1] for core in tt_cores]
    return tensor.reshape(dimensions)


def tt_svd(tensor, rank):
    """
    Perform TT-SVD decomposition of a full tensor.

    Args:
        tensor (np.ndarray): Input tensor of shape (d1, d2, ..., dn)
        rank (int or list): TT-ranks; if int, all intermediate ranks are set to this value

    Returns:
        list of np.ndarray: List of TT cores [G1, G2, ..., Gn]
    """
    shape = list(tensor.shape)
    n = len(shape)  # Number of dimensions

    # Handle rank input
    if isinstance(rank, int):
        rank = [rank] * n
    elif len(rank) != n:
        raise ValueError("Length of rank must match number of tensor dimensions.")

    # Prepend and append rank 1 at start/end
    tt_ranks = [1] + rank + [1]

    cores = []
    unfolding = tensor.copy()

    for i in range(n):
        if i==n-1:
            break
        curr_dim = shape[i]
        next_rank = tt_ranks[i + 1]

        # Reshape into matrix
        unfolding = unfolding.reshape(tt_ranks[i] * curr_dim, -1)

        # SVD
        U, S, Vt = np.linalg.svd(unfolding, full_matrices=False)

        # Truncate based on desired rank
        U_trunc = U[:, :next_rank]
        S_trunc = S[:next_rank]
        Vt_trunc = Vt[:next_rank, :]

        # Reshape U into TT-core
        core = U_trunc.reshape(tt_ranks[i], curr_dim, next_rank)
        cores.append(core)

        # Update unfolding with S @ Vt
        unfolding = np.diag(S_trunc) @ Vt_trunc
    unfolding = unfolding.reshape(tt_ranks[-2], shape[-1], tt_ranks[-1])
    cores.append(unfolding)
    return cores


def left_orthogonalize(tt_cores):
    d = len(tt_cores)

    for k in range(d - 1):
        # Reshape current core into a tall matrix: (r_prev * n_k) x r_curr
        G = tt_cores[k]
        r_prev, n_k, r_curr = G.shape
        Gmat = G.reshape(r_prev * n_k, r_curr)

        # QR decomposition to orthogonalize
        if k!=d-1:
            U, S, Vt = np.linalg.svd(Gmat,full_matrices=False)
            Q = U
            R = np.diag(S) @ Vt
            # Update current core with orthogonal part
            tt_cores[k] = U.reshape(r_prev, n_k, U.shape[1])
        # Multiply R into the next core
            G_next = tt_cores[k + 1]
            r_next_prev, n_next, r_next_curr = G_next.shape
            # Reshape next core to apply R on the left
            G_next_reshaped = G_next.reshape(r_next_prev, n_next, r_next_curr)

            G_next_reshaped = np.einsum('ij,jkl->ikl', R, G_next_reshaped)
            r_next_prev, n_next, r_next_curr = G_next_reshaped.shape
            tt_cores[k + 1] = G_next_reshaped

        else:
            Q = Gmat

            tt_cores[k] = Gmat

        print(f"tt_cores{k}.shape: {tt_cores[k].shape}")
    print(f"tt_cores[-1].shape: {tt_cores[-1].shape}")
    return tt_cores


def right_orthogonalize(tt_cores):
    d = len(tt_cores)
    for k in reversed(range(1, d)):
        G = tt_cores[k]
        r_prev, n_k, r_curr = G.shape
        # Reshape current core into a wide matrix: r_prev x (n_k * r_curr)
        Gmat = G.reshape(r_prev, n_k * r_curr)
        # SVD
        U, S, Vt = np.linalg.svd(Gmat, full_matrices=False)
        # Vt: (rank, n_k * r_curr)
        # U: (r_prev, rank)
        # S: (rank,)
        rank = Vt.shape[0]
        # Update current core with right-orthogonal part
        tt_cores[k] = Vt.reshape(rank, n_k, r_curr)
        # Multiply U @ diag(S) into the previous core
        G_prev = tt_cores[k-1]
        r_prev_prev, n_prev, r_prev_curr = G_prev.shape
        G_prev_mat = G_prev.reshape(r_prev_prev * n_prev, r_prev_curr)
        # (r_prev_prev * n_prev, r_prev_curr) @ (r_prev_curr, rank)
        G_prev_new = G_prev_mat @ (U @ np.diag(S))
        tt_cores[k-1] = G_prev_new.reshape(r_prev_prev, n_prev, rank)
        print(f"tt_cores[{k}].shape: {tt_cores[k].shape}")
    print(f"tt_cores[0].shape: {tt_cores[0].shape}")
    return tt_cores


def tt_orthogonalize(tt_cores, types="svd", left_right="left"):
    '''
    inputs: tt_cores: list of numpy arrays or list of torch tensors
    args: types: "qr" or "svd"
          left_right: "left" or "right"
    outputs: tt_cores: list of numpy arrays or list of torch tensors
    '''

    if left_right == "left":
        d = len(tt_cores)
        for k in range(d - 1):
            # Reshape current core into a tall matrix: (r_prev * n_k) x r_curr
            G = tt_cores[k]
            r_prev, n_k, r_curr = G.shape
            Gmat = G.reshape(r_prev * n_k, r_curr)

            # QR decomposition to orthogonalize

            if k != d - 1:
                if types == "svd":

                    U, S, Vt = np.linalg.svd(Gmat, full_matrices=False)
                    Q = U
                    R = np.diag(S) @ Vt
                elif types == "qr":
                    Q, R = np.linalg.qr(Gmat)
                    U = Q
                else:
                    raise ValueError("Invalid type. Choose 'qr' or 'svd'.")
                # Update current core with orthogonal part
                tt_cores[k] = U.reshape(r_prev, n_k, U.shape[1])
            # Multiply R into the next core
                G_next = tt_cores[k + 1]
                r_next_prev, n_next, r_next_curr = G_next.shape
                # Reshape next core to apply R on the left
                G_next_reshaped = G_next.reshape(r_next_prev, n_next, r_next_curr)

                G_next_reshaped = np.einsum('ij,jkl->ikl', R, G_next_reshaped)
                r_next_prev, n_next, r_next_curr = G_next_reshaped.shape
                tt_cores[k + 1] = G_next_reshaped

            else:
                Q = Gmat

                tt_cores[k] = Gmat
    elif left_right == "right":
        d = len(tt_cores)
        for k in reversed(range(1, d)):
            G = tt_cores[k]
            r_prev, n_k, r_curr = G.shape
            # Reshape current core into a wide matrix: r_prev x (n_k * r_curr)
            Gmat = G.reshape(r_prev, n_k * r_curr)
            # SVD
            if types == "svd":
                U, S, Vt = np.linalg.svd(Gmat, full_matrices=False)
                # Vt: (rank, n_k * r_curr)
                # U: (r_prev, rank)
                # S: (rank,)
                rank = Vt.shape[0]
                # Update current core with right-orthogonal part
                tt_cores[k] = Vt.reshape(rank, n_k, r_curr)
                # Multiply U @ diag(S) into the previous core
                G_prev = tt_cores[k-1]
                r_prev_prev, n_prev, r_prev_curr = G_prev.shape
                G_prev_mat = G_prev.reshape(r_prev_prev * n_prev, r_prev_curr)
                # (r_prev_prev * n_prev, r_prev_curr) @ (r_prev_curr, rank)
                G_prev_new = G_prev_mat @ (U @ np.diag(S))
                tt_cores[k-1] = G_prev_new.reshape(r_prev_prev, n_prev, rank)
            elif types == "qr":
                # # QR decomposition
                # Q, R = np.linalg.qr(Gmat)


                # # Q: (r_prev, n_k * r_curr)
                # # R: (n_k * r_curr, n_k * r_curr)
                # # Update current core with right-orthogonal part


                # tt_cores[k] = Q.reshape(r_prev, n_k, r_curr)
                # # Multiply R into the previous core
                # G_prev = tt_cores[k-1]
                # r_prev_prev, n_prev, r_prev_curr = G_prev.shape
                # G_prev_mat = G_prev.reshape(r_prev_prev * n_prev, r_prev_curr)



                # # (r_prev_prev * n_prev, r_prev_curr) @ (r_prev_curr, n_k * r_curr)
                # G_prev_new = G_prev_mat @ R
                # tt_cores[k-1] = G_prev_new.reshape(r_prev_prev, n_prev, n_k * r_curr)
                raise NotImplementedError("Right orthogonalization with QR decomposition is not implemented.")
            else:
                raise ValueError("Invalid orthogonalization type. Choose 'svd' or 'qr'.")

    return tt_cores


def tt_orth_at(x, pos, dir):
    """Orthogonalize single core.

    x = orth_at(x, pos, 'left') left-orthogonalizes the core at position pos
    and multiplies the corresponding R-factor with core pos+1. All other cores
    are untouched. The modified tensor is returned.

    x = orth_at(x, pos, 'right') right-orthogonalizes the core at position pos
    and multiplies the corresponding R-factor with core pos-1. All other cores
    are untouched. The modified tensor is returned.

    See also orthogonalize.
    """

    # Adapted from the TTeMPS Toolbox.

    Un = x[pos] # get the core at position pos
    if dir.lower() == 'left':
        Q, R = tl.qr(Un.reshape(-1, x.rank[pos+1]), mode='reduced') # perform QR decomposition
        # Fixed signs of x.U{pos},  if it is orthogonal.
        # This needs for the ASCU algorithm when it updates ranks both sides.
        sR = np.sign(np.diag(R)) # get the signs of the diagonal elements of R
        Q = Q * sR # multiply Q by the signs
        R = (R.T * sR).T # multiply R by the signs

        # Note that orthogonalization might change the ranks of the core Xn
        # and X{n+1}. For such case, the number of entries is changed
        # accordingly.
        # Need to change structure of the tt-tensor
        # pos(n+1)
        #
        # update the core X{n}
        Un = Q.reshape(x.rank[pos], x.shape[pos], -1) # reshape Q to the core shape

        # update the core X{n+1}
        Un2 = R @ x[pos+1].reshape(x.rank[pos+1], -1) # multiply R by the next core

        # Check if rank-n is preserved
        x[pos] = Un.reshape(x[pos].shape[0],x[pos].shape[1],R.shape[0]) # update the current core
        x[pos+1] = Un2.reshape(R.shape[0],x[pos+1].shape[1],-1) # update the next core

        if R.shape[0] != x.rank[pos+1]:
            x.rank = list(x.rank) # convert the tuple to a list
            x.rank[pos+1] = R.shape[0] # assign the new value
            x.rank = tuple(x.rank) # convert the list back to a tuple
    elif dir.lower() == 'right':
        # mind the transpose as we want to orthonormalize rows
        Q, R = tl.qr(Un.reshape(x.rank[pos], -1).T, mode='reduced') # perform QR decomposition on the transpose
        # Fixed signs of x.U{pos},  if it is orthogonal.
        # This needs for the ASCU algorithm when it updates ranks both
        # sides.
        sR = np.sign(np.diag(R)) # get the signs of the diagonal elements of R
        Q = Q * sR # multiply Q by the signs
        R = (R.T * sR).T # multiply R by the signs

        Un = Q.T.reshape(-1, x.shape[pos], x.rank[pos+1]) # reshape Q transpose to the core shape
        Un2 = x[pos-1].reshape(-1, x.rank[pos]) @  R.T # multiply the previous core by R transpose

        x[pos] = Un.reshape(Un.shape[0],x[pos].shape[1],-1) # update the current core
        x[pos-1] = Un2.reshape(x[pos-1].shape[0],x[pos-1].shape[1],-1) # update the previous core

        if R.shape[0] != x.rank[pos]:
            x.rank = list(x.rank) # convert the tuple to a list
            x.rank[pos] = R.shape[0] # assign the new value
            x.rank = tuple(x.rank) # convert the list back to a tuple
    else:
        raise ValueError('Unknown direction specified. Choose either LEFT or RIGHT')

    return x


def tt_orthogonalize(x, pos):
    """Orthogonalize tensor.

    x = orthogonalize(x, pos) orthogonalizes all cores of the TTeMPS tensor x
    except the core at position pos. Cores 1...pos-1 are left-, cores pos+1...end
    are right-orthogonalized. Therefore,

    x = orthogonalize(x, 1) right-orthogonalizes the full tensor,

    x = orthogonalize(x, x.order) left-orthogonalizes the full tensor.

    See also orth_at.
    """

    # adapted from the TTeMPS Toolbox.

    # left orthogonalization till pos (from left)
    for i in range(pos):
        # print(f'Left orthogonalization {i}')
        x = tt_orth_at(x, i, 'left')

    # right orthogonalization till pos (from right)
    ndimX = len(x.factors)

    for i in range(ndimX-1, pos, -1):
        # print(f'Right orthogonalization {i}')
        x = tt_orth_at(x, i, 'right')

    return x


def is_left_orthogonal(core):
    r_prev, n, r_curr = core.shape
    Gmat = core.reshape(r_prev * n, r_curr)
    return np.allclose(Gmat.T @ Gmat, np.eye(r_curr))


def is_right_orthogonal(core):
    r_prev, n, r_curr = core.shape
    Gmat = core.reshape(r_prev, n * r_curr)
    return np.allclose(Gmat @ Gmat.T, np.eye(r_prev))


def DMD(X1, X2, r, dt):
        U, s, Vh = np.linalg.svd(X1, full_matrices=False)
        Ur = U[:, :r]
        Sr = np.diag(s[:r])
        Vr = Vh.conj().T[:, :r]

        Atilde = Ur.conj().T @ X2 @ Vr @ np.linalg.inv(Sr)
        Lambda, W = np.linalg.eig(Atilde)

        Phi = X2 @ Vr @ np.linalg.inv(Sr) @ W
        omega = np.log(Lambda) / dt

        alpha1 = np.linalg.lstsq(Phi, X1[:, 0], rcond=None)[0]
        b = np.linalg.lstsq(Phi, X2[:, 0], rcond=None)[0]

        time_dynamics = None
        for i in range(X1.shape[1]):
            v = np.array(alpha1)[:, 0] * np.exp(np.array(omega) * (i + 1) * dt)
            if time_dynamics is None:
                time_dynamics = v
            else:
                time_dynamics = np.vstack((time_dynamics, v))
        X_dmd = np.dot(np.array(Phi), time_dynamics.T)

        return Phi, omega, Lambda, alpha1, b, X_dmd, time_dynamics.T


def matricize_tt(tt_cores, row_modes, col_modes, dims):
    """
    Matricizes a Tensor Train (TT) tensor into a matrix.

    Args:
        tt_cores (list of numpy.ndarray): List of TT-cores.
        row_modes (tuple of int): Indices of modes to form rows.
        col_modes (tuple of int): Indices of modes to form columns.
        dims (tuple of int): Original dimensions of the tensor.

    Returns:
        numpy.ndarray: Matricized tensor.
    """
    num_cores = len(tt_cores)
    row_size = np.prod([dims[i] for i in row_modes])
    col_size = np.prod([dims[i] for i in col_modes])

    # Initial contraction
    core_idx = 0
    if core_idx in row_modes:
        row_pos = row_modes.index(core_idx)
        curr_rows = dims[core_idx]
    else:
        row_pos = -1
        curr_rows = 1

    if core_idx in col_modes:
      col_pos = col_modes.index(core_idx)
      curr_cols = dims[core_idx]
    else:
      col_pos = -1
      curr_cols = 1

    matrix = tt_cores[core_idx]

    # Subsequent contractions
    for core_idx in range(1, num_cores):
        if core_idx in row_modes:
          row_pos = row_modes.index(core_idx)
          next_rows = dims[core_idx]
        else:
          row_pos = -1
          next_rows = 1

        if core_idx in col_modes:
          col_pos = col_modes.index(core_idx)
          next_cols = dims[core_idx]
        else:
          col_pos = -1
          next_cols = 1

        if row_pos != -1 and col_pos != -1:
          matrix = np.einsum('aij,jbk->abik', matrix, tt_cores[core_idx])
          matrix = matrix.reshape(curr_rows*next_rows, curr_cols*next_cols, matrix.shape[-1])
          curr_rows *= next_rows
          curr_cols *= next_cols
        elif row_pos != -1:
          matrix = np.einsum('aij,jbk->abik', matrix, tt_cores[core_idx])
          matrix = matrix.reshape(curr_rows*next_rows, matrix.shape[-1])
          curr_rows *= next_rows
        elif col_pos != -1:
          matrix = np.einsum('ai,ijk->ajk', matrix, tt_cores[core_idx])
          matrix = matrix.reshape(matrix.shape[0], curr_cols*next_cols)
          curr_cols *= next_cols
        else:
          matrix = np.einsum('ai,ijk->ajk', matrix, tt_cores[core_idx])

    return matrix.reshape(row_size, col_size)


def pseudoinverse(tt_cores: list, l: int):
    """Calculate pseudoinverse.

    Parameters
    ----------
        tt_cores : list
            List of TT cores, each is a np.ndarray with a shape of (r_prev,
            d_mode, r_next).
        l : int
            Core number to split the tensor train and assemble a matrix.

    Returns
    -------
        Pseudoinverse matrix of tt decomposition relative to given core number.
    """

    # Step 1. Given a tensor x in TT-format and core number 1 <= l <= d − 1 to
    # compute pseudoinverse of x(n_1, ..., n_l; n_(l+1), ..., n_d).

    tt_cores_copy = np.copy(tt_cores)
    d = tt_cores_copy.shape[0]

    if not 1 <= l <= d-1:
        raise ValueError("""Core number must be between 1 and d-1. It is not
                            possible to split a tensor train otherwise.""")

    l -= 1

    # Step 2. Left-orthonormalize x_1, ..., x_(l-1) and right-orthonormalize
    # x_d, ..., x_(l+1).

    tt_cores_copy[0:l-1] = left_orthogonalize(tt_cores_copy[0:l-1])
    tt_cores_copy[l+1:d:-1] = right_orthogonalize(tt_cores_copy[l+1:d:-1])

    # Step 3. Compute SVD of x_l(r_(l-1), n_l; r_l).

    x_l = tt_cores[l]
    r_l_prev, n_l, r_l = x_l.shape
    x_l_mat = x_l.reshape(r_l_prev * n_l, r_l)

    U, S, V_T = np.linalg.svd(x_l_mat, full_matrices=False)

    # Step 4. Define tensor "y".

    y = U.reshape(r_l_prev, n_l, U.shape[1])

    # Step 5. Define tensor "z".

    x_l_next = tt_cores[l]
    r_l, n_l_next, r_l_next = x_l_next.shape
    x_l_next_mat = x_l_next.reshape(r_l, n_l_next * r_l_next)

    z = V_T @ x_l_next_mat

    # Step 6.

    tt_cores_copy[l] = y
    tt_cores_copy[l+1] = z

    # Step 7. Compute matrix M.

    k_l_prev = tt_cores_copy[l-1].shape[1]
    M = np.tensordot(tt_cores_copy[0:l-1], tt_cores_copy[l,k_l_prev,:,:], axes=([-1], [0]))
    n_prod_left = 1
    for i in range(l):
        n_prod_left *= tt_cores_copy[i].shape[2]
    M = M.reshape(n_prod_left, r_l)

    # Step 8. Compute matrix N.

    k_l_next = tt_cores_copy[l+1].shape[1]
    N = np.tensordot(tt_cores_copy[l+1,:,:,k_l_next], tt_cores_copy[l+2:], axes=([-1], [0]))
    n_prod_right = 1
    for i in range(l+1, d):
        n_prod_left *= tt_cores_copy[i].shape[2]
    N = N.reshape(r_l, n_prod_right)

    # Step 9. Compute pseudo-inverse and return.

    return N.T @ np.inv(Sigma) @ M.T
