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


# def dynamic_mode_decomposition(xi, t, f, r=20,indices=[170, 199, 210]):
#     """
#     Perform Dynamic Mode Decomposition (DMD) on spatio-temporal data.

#     Parameters:
#         xi (np.ndarray): Spatial grid points.
#         t (np.ndarray): Time points.
#         f (np.ndarray): Spatio-temporal data matrix of shape (len(t), len(xi)).
#         r (int): Rank for SVD truncation.
#     """
    
#     N_train = f.shape[0] // 2
#     f_train = f[:N_train]
#     t_train = t[:N_train]

#     # Step 1: Build linear system
#     X = f_train.T
#     X1 = X[:, :-1]
#     X2 = X[:, 1:]

#     # Step 2: Singular Value Decomposition
#     U, Sdiag, Vh = np.linalg.svd(X1, full_matrices=False)
#     S = np.diag(Sdiag)
#     V = np.conj(Vh).T

#     Ur = U[:, :r]
#     Sr = S[:r, :r]
#     Vr = V[:, :r]
#     # Step 3: Compute Atilde
#     Atilde = np.dot(np.conj(Ur.T), np.dot(X2, np.dot(Vr, np.linalg.inv(Sr))))
#     Lambda_diag, W = np.linalg.eig(Atilde)
#     Phi = np.dot(X2, np.dot(Vr, np.dot(linalg.inv(Sr), W)))

#     omega = np.log(Lambda_diag) / (t[1] - t[0])

#     # Step 4: Reconstruct signal
#     x1 = X[:, 0]
#     b = np.linalg.pinv(Phi) @ x1

#     t_dyn = np.zeros((r, len(t_train)), dtype=complex)
#     for i in range(len(t_train)):
#         t_dyn[:, i] = b * np.exp(omega * t_train[i])
#     f_dmd = Phi @ t_dyn

#     # Step 5: Predict future dynamics
#     t_ext = t
#     t_ext_dyn = np.zeros((r, len(t_ext)), dtype=complex)
#     for i in range(len(t_ext)):
#         t_ext_dyn[:, i] = b * np.exp(omega * t_ext[i])
#     f_dmd_ext = Phi @ t_ext_dyn
#     dmd = pydmd.DMD(svd_rank=r)
#     dmd.fit(f)

#     plot_singular_values(Sdiag)
#     plot_dynamic_modes(xi, Phi)
#     plot_eigenvalues(Lambda_diag)
#     plot_time_modes(t_train, t_dyn)
#     plot_original_vs_reconstructed_1d(t_train, xi, f_train, f_dmd, indices=indices)
#     plot_2d_comparison(t_train, xi, f_train, f_dmd)
#     plot_prediction(t, t_train, xi, f, f_dmd_ext, N_train, indices=indices)
#     plot_2d_prediction(t_ext, xi, f_dmd_ext)
#     return dmd

# # Helper Plotting Functions

# def plot_singular_values(S):
#     plt.figure(figsize=(6, 6))
#     plt.plot(S, ".")

#     plt.yscale("log")
#     plt.title("Singular Values (log scale)")
#     plt.xlabel("Index")
#     plt.ylabel("Singular Value")
#     plt.grid(True)
#     plt.show()


# def plot_dynamic_modes(xi, Phi):
#     plt.figure(figsize=(16, 6))
#     plt.title("Dynamic Modes")
#     for i in range(min(3, Phi.shape[1])):
#         plt.plot(xi, np.real(Phi[:, i]), label=f'$\Phi_{i}$')
#     plt.legend()
#     plt.xlabel("x")
#     plt.ylabel("Mode amplitude")
#     plt.grid(True)
#     plt.show()


# def plot_eigenvalues(Lambda_diag):
#     theta = np.linspace(0, 2 * np.pi, 150)
#     radius = 1
#     a = radius * np.cos(theta)
#     b = radius * np.sin(theta)

#     plt.figure(figsize=(6, 6))
#     for i in range(min(3, len(Lambda_diag))):
#         plt.scatter(np.real(Lambda_diag[i]), np.imag(Lambda_diag[i]), label=f"$\lambda_{i}$")
#     plt.plot(a, b, color='k', ls='--', label="Unit circle")
#     plt.xlabel("$\lambda_r$")
#     plt.ylabel("$\lambda_i$")
#     plt.legend()
#     plt.grid(True)
#     plt.axis("equal")
#     plt.title("Eigenvalues of $\\tilde{A}$")
#     plt.show()


# def plot_time_modes(t_train, t_dyn):
#     plt.figure(figsize=(16, 6))
#     plt.title("Time Modes")
#     for i in range(min(3, t_dyn.shape[0])):
#         plt.plot(t_train, np.real(t_dyn[i, :]), label=f'Time mode {i}')
#     plt.legend()
#     plt.xlabel("Time")
#     plt.ylabel("Amplitude")
#     plt.grid(True)
#     plt.show()


# def plot_original_vs_reconstructed_1d(t_train, xi, f_train, f_dmd, indices):
#     for idx in indices:
#         plt.figure(figsize=(16, 6))
#         plt.title(f"1D Plot at x = {idx}")
#         plt.plot(t_train, np.real(f_train[:, idx]), label='Original data')
#         plt.plot(t_train, np.real(f_dmd[idx, :]), '--', label='Reconstructed by DMD')
#         plt.legend()
#         plt.xlabel("Time")
#         plt.ylabel("f(x,t)")
#         plt.grid(True)
#         plt.show()


# def plot_2d_comparison(t_train, xi, f_train, f_dmd):
#     plt.figure(figsize=(16, 6))
#     plt.title('2D Plot: Original Data')
#     plt.pcolormesh(t_train, xi, np.real(f_train.T), shading='auto')
#     plt.colorbar(label='f(x,t)')
#     plt.xlabel("Time")
#     plt.ylabel("Space (x)")
#     plt.show()

#     plt.figure(figsize=(16, 6))
#     plt.title('2D Plot: DMD Reconstruction')
#     plt.pcolormesh(t_train, xi, np.real(f_dmd), shading='auto')
#     plt.colorbar(label='f(x,t)')
#     plt.xlabel("Time")
#     plt.ylabel("Space (x)")
#     plt.show()


# def plot_prediction(t, t_train, xi, f, f_dmd_ext, N_train, indices):
#     for idx in indices:
#         plt.figure(figsize=(16, 6))
#         plt.title(f"Prediction at x = {idx}")
#         plt.plot(t, np.real(f[:, idx]), label='Original data')
#         plt.plot(t_train, np.real(f_dmd_ext[idx, :N_train]), '--', label='Reconstructed by DMD')
#         plt.plot(t[N_train:], np.real(f_dmd_ext[idx, N_train:]), '.', label='Predicted by DMD')
#         plt.legend()
#         plt.xlabel("Time")
#         plt.ylabel("f(x,t)")
#         plt.grid(True)
#         plt.show()


# def plot_2d_prediction(t_ext, xi, f_dmd_ext):
#     plt.figure(figsize=(16, 6))
#     plt.title('2D Plot: Extended Prediction')
#     plt.pcolormesh(t_ext, xi, np.real(f_dmd_ext), shading='auto')
#     plt.colorbar(label='f(x,t)')
#     plt.xlabel("Time")
#     plt.ylabel("Space (x)")
#     plt.show()


# def delayed_matrix(X, delay,vstack=True):
#     n, t = X.shape
#     cols = t - delay + 1
#     X_delayed = [X[:, i:i+cols] for i in range(delay)]
#     if not vstack:
#         return np.hstack(X_delayed)
#     return np.vstack(X_delayed)

# def tolerance(S, threshold=1e-6):
#     S_squared = S**2
#     total_energy = np.sum(S_squared)
#     cumulative = np.cumsum(S_squared[::-1])[::-1]
#     EE = cumulative / total_energy
#     N = np.argmax(EE <= threshold)
#     return int(N)

# def HODMD(X, delay, energy_threshold, dt):
#     X_aug = delayed_matrix(X, delay)
#     X1 = X_aug[:, :-1]
#     X2 = X_aug[:, 1:]

#     U, S, Vh = np.linalg.svd(X1, full_matrices=False)
#     r = tolerance(S, energy_threshold)

#     Ur = U[:, :r].astype(np.complex128)
#     Sr = np.diag(S[:r]).astype(np.complex128)
#     Vr = Vh.conj().T[:, :r].astype(np.complex128)

#     Atilde = Ur.conj().T @ X2 @ Vr @ np.linalg.inv(Sr)
#     Lambda, W = np.linalg.eig(Atilde)
#     Phi = X2 @ Vr @ np.linalg.inv(Sr) @ W
#     omega = np.log(Lambda) / dt

#     alpha1 = np.linalg.lstsq(Phi, X1[:, 0], rcond=None)[0]

#     time_dynamics = np.zeros((r, X1.shape[1]), dtype=np.complex128)
#     print("x1shape",X1.shape[1])
#     for i in range(X1.shape[1]):
#         time_dynamics[:, i] = alpha1 * np.exp(omega * ((i + 1) * dt))

#     X_dmd = Phi @ time_dynamics
#     return X_dmd, r

# def reconstruct_from_tt(tt_cores):
#     """
#     Reconstructs a full tensor from its TT decomposition.

#     Args:
#         tt_cores (list of np.ndarray): List of TT cores. Each core has shape (r_prev, d_mode, r_next).

#     Returns:
#         np.ndarray: Reconstructed full tensor.
#     """
#     tensor = tt_cores[0]  # Start with first core
#     for i in range(1, len(tt_cores)):
#         # Contract along the bond dimension
#         tensor = np.tensordot(tensor, tt_cores[i], axes=([-1], [0]))

#     # Get original dimensions from the cores
#     dimensions = [core.shape[1] for core in tt_cores]
#     return tensor.reshape(dimensions)


# def tt_svd(tensor, rank):
#     """
#     Perform TT-SVD decomposition of a full tensor.

#     Args:
#         tensor (np.ndarray): Input tensor of shape (d1, d2, ..., dn)
#         rank (int or list): TT-ranks; if int, all intermediate ranks are set to this value

#     Returns:
#         list of np.ndarray: List of TT cores [G1, G2, ..., Gn]
#     """
#     shape = list(tensor.shape)
#     n = len(shape)  # Number of dimensions

#     # Handle rank input
#     if isinstance(rank, int):
#         rank = [rank] * n
#     elif len(rank) != n:
#         raise ValueError("Length of rank must match number of tensor dimensions.")

#     # Prepend and append rank 1 at start/end
#     tt_ranks = [1] + rank + [1]

#     cores = []
#     unfolding = tensor.copy()

#     for i in range(n):
#         if i==n-1:
#             break
#         curr_dim = shape[i]
#         next_rank = tt_ranks[i + 1]

#         # Reshape into matrix
#         unfolding = unfolding.reshape(tt_ranks[i] * curr_dim, -1)

#         # SVD
#         U, S, Vt = np.linalg.svd(unfolding, full_matrices=False)

#         # Truncate based on desired rank
#         U_trunc = U[:, :next_rank]
#         S_trunc = S[:next_rank]
#         Vt_trunc = Vt[:next_rank, :]

#         # Reshape U into TT-core
#         core = U_trunc.reshape(tt_ranks[i], curr_dim, next_rank)
#         cores.append(core)

#         # Update unfolding with S @ Vt
#         unfolding = np.diag(S_trunc) @ Vt_trunc
#     unfolding = unfolding.reshape(tt_ranks[-2], shape[-1], tt_ranks[-1])
#     cores.append(unfolding)
#     return cores


# def left_orthogonalize(tt_cores):
#     d = len(tt_cores)

#     for k in range(d - 1):
#         # Reshape current core into a tall matrix: (r_prev * n_k) x r_curr
#         G = tt_cores[k]
#         r_prev, n_k, r_curr = G.shape
#         Gmat = G.reshape(r_prev * n_k, r_curr)

#         # QR decomposition to orthogonalize
#         if k!=d-1:
#             U, S, Vt = np.linalg.svd(Gmat,full_matrices=False)
#             Q = U
#             R = np.diag(S) @ Vt
#             # Update current core with orthogonal part
#             tt_cores[k] = U.reshape(r_prev, n_k, U.shape[1])
#         # Multiply R into the next core
#             G_next = tt_cores[k + 1]
#             r_next_prev, n_next, r_next_curr = G_next.shape
#             # Reshape next core to apply R on the left
#             G_next_reshaped = G_next.reshape(r_next_prev, n_next, r_next_curr)

#             G_next_reshaped = np.einsum('ij,jkl->ikl', R, G_next_reshaped)
#             r_next_prev, n_next, r_next_curr = G_next_reshaped.shape
#             tt_cores[k + 1] = G_next_reshaped

#         else:
#             Q = Gmat

#             tt_cores[k] = Gmat

#         print(f"tt_cores{k}.shape: {tt_cores[k].shape}")
#     print(f"tt_cores[-1].shape: {tt_cores[-1].shape}")
#     return tt_cores


# def right_orthogonalize(tt_cores):
#     d = len(tt_cores)
#     for k in reversed(range(1, d)):
#         G = tt_cores[k]
#         r_prev, n_k, r_curr = G.shape
#         # Reshape current core into a wide matrix: r_prev x (n_k * r_curr)
#         Gmat = G.reshape(r_prev, n_k * r_curr)
#         # SVD
#         U, S, Vt = np.linalg.svd(Gmat, full_matrices=False)
#         # Vt: (rank, n_k * r_curr)
#         # U: (r_prev, rank)
#         # S: (rank,)
#         rank = Vt.shape[0]
#         # Update current core with right-orthogonal part
#         tt_cores[k] = Vt.reshape(rank, n_k, r_curr)
#         # Multiply U @ diag(S) into the previous core
#         G_prev = tt_cores[k-1]
#         r_prev_prev, n_prev, r_prev_curr = G_prev.shape
#         G_prev_mat = G_prev.reshape(r_prev_prev * n_prev, r_prev_curr)
#         # (r_prev_prev * n_prev, r_prev_curr) @ (r_prev_curr, rank)
#         G_prev_new = G_prev_mat @ (U @ np.diag(S))
#         tt_cores[k-1] = G_prev_new.reshape(r_prev_prev, n_prev, rank)
#         print(f"tt_cores[{k}].shape: {tt_cores[k].shape}")
#     print(f"tt_cores[0].shape: {tt_cores[0].shape}")
#     return tt_cores


# def tt_orthogonalize(tt_cores, types="svd", left_right="left"):
#     '''
#     inputs: tt_cores: list of numpy arrays or list of torch tensors
#     args: types: "qr" or "svd"
#           left_right: "left" or "right"
#     outputs: tt_cores: list of numpy arrays or list of torch tensors
#     '''

#     if left_right == "left":
#         d = len(tt_cores)
#         for k in range(d - 1):
#             # Reshape current core into a tall matrix: (r_prev * n_k) x r_curr
#             G = tt_cores[k]
#             r_prev, n_k, r_curr = G.shape
#             Gmat = G.reshape(r_prev * n_k, r_curr)

#             # QR decomposition to orthogonalize

#             if k != d - 1:
#                 if types == "svd":

#                     U, S, Vt = np.linalg.svd(Gmat, full_matrices=False)
#                     Q = U
#                     R = np.diag(S) @ Vt
#                 elif types == "qr":
#                     Q, R = np.linalg.qr(Gmat)
#                     U = Q
#                 else:
#                     raise ValueError("Invalid type. Choose 'qr' or 'svd'.")
#                 # Update current core with orthogonal part
#                 tt_cores[k] = U.reshape(r_prev, n_k, U.shape[1])
#             # Multiply R into the next core
#                 G_next = tt_cores[k + 1]
#                 r_next_prev, n_next, r_next_curr = G_next.shape
#                 # Reshape next core to apply R on the left
#                 G_next_reshaped = G_next.reshape(r_next_prev, n_next, r_next_curr)

#                 G_next_reshaped = np.einsum('ij,jkl->ikl', R, G_next_reshaped)
#                 r_next_prev, n_next, r_next_curr = G_next_reshaped.shape
#                 tt_cores[k + 1] = G_next_reshaped

#             else:
#                 Q = Gmat

#                 tt_cores[k] = Gmat
#     elif left_right == "right":
#         d = len(tt_cores)
#         for k in reversed(range(1, d)):
#             G = tt_cores[k]
#             r_prev, n_k, r_curr = G.shape
#             # Reshape current core into a wide matrix: r_prev x (n_k * r_curr)
#             Gmat = G.reshape(r_prev, n_k * r_curr)
#             # SVD
#             if types == "svd":
#                 U, S, Vt = np.linalg.svd(Gmat, full_matrices=False)
#                 # Vt: (rank, n_k * r_curr)
#                 # U: (r_prev, rank)
#                 # S: (rank,)
#                 rank = Vt.shape[0]
#                 # Update current core with right-orthogonal part
#                 tt_cores[k] = Vt.reshape(rank, n_k, r_curr)
#                 # Multiply U @ diag(S) into the previous core
#                 G_prev = tt_cores[k-1]
#                 r_prev_prev, n_prev, r_prev_curr = G_prev.shape
#                 G_prev_mat = G_prev.reshape(r_prev_prev * n_prev, r_prev_curr)
#                 # (r_prev_prev * n_prev, r_prev_curr) @ (r_prev_curr, rank)
#                 G_prev_new = G_prev_mat @ (U @ np.diag(S))
#                 tt_cores[k-1] = G_prev_new.reshape(r_prev_prev, n_prev, rank)
#             elif types == "qr":
#                 # # QR decomposition
#                 # Q, R = np.linalg.qr(Gmat)


#                 # # Q: (r_prev, n_k * r_curr)
#                 # # R: (n_k * r_curr, n_k * r_curr)
#                 # # Update current core with right-orthogonal part


#                 # tt_cores[k] = Q.reshape(r_prev, n_k, r_curr)
#                 # # Multiply R into the previous core
#                 # G_prev = tt_cores[k-1]
#                 # r_prev_prev, n_prev, r_prev_curr = G_prev.shape
#                 # G_prev_mat = G_prev.reshape(r_prev_prev * n_prev, r_prev_curr)



#                 # # (r_prev_prev * n_prev, r_prev_curr) @ (r_prev_curr, n_k * r_curr)
#                 # G_prev_new = G_prev_mat @ R
#                 # tt_cores[k-1] = G_prev_new.reshape(r_prev_prev, n_prev, n_k * r_curr)
#                 raise NotImplementedError("Right orthogonalization with QR decomposition is not implemented.")
#             else:
#                 raise ValueError("Invalid orthogonalization type. Choose 'svd' or 'qr'.")

#     return tt_cores


# def tt_orth_at(x, pos, dir):
#     """Orthogonalize single core.

#     x = orth_at(x, pos, 'left') left-orthogonalizes the core at position pos
#     and multiplies the corresponding R-factor with core pos+1. All other cores
#     are untouched. The modified tensor is returned.

#     x = orth_at(x, pos, 'right') right-orthogonalizes the core at position pos
#     and multiplies the corresponding R-factor with core pos-1. All other cores
#     are untouched. The modified tensor is returned.

#     See also orthogonalize.
#     """

#     # Adapted from the TTeMPS Toolbox.

#     Un = x[pos] # get the core at position pos
#     if dir.lower() == 'left':
#         Q, R = tl.qr(Un.reshape(-1, x.rank[pos+1]), mode='reduced') # perform QR decomposition
#         # Fixed signs of x.U{pos},  if it is orthogonal.
#         # This needs for the ASCU algorithm when it updates ranks both sides.
#         sR = np.sign(np.diag(R)) # get the signs of the diagonal elements of R
#         Q = Q * sR # multiply Q by the signs
#         R = (R.T * sR).T # multiply R by the signs

#         # Note that orthogonalization might change the ranks of the core Xn
#         # and X{n+1}. For such case, the number of entries is changed
#         # accordingly.
#         # Need to change structure of the tt-tensor
#         # pos(n+1)
#         #
#         # update the core X{n}
#         Un = Q.reshape(x.rank[pos], x.shape[pos], -1) # reshape Q to the core shape

#         # update the core X{n+1}
#         Un2 = R @ x[pos+1].reshape(x.rank[pos+1], -1) # multiply R by the next core

#         # Check if rank-n is preserved
#         x[pos] = Un.reshape(x[pos].shape[0],x[pos].shape[1],R.shape[0]) # update the current core
#         x[pos+1] = Un2.reshape(R.shape[0],x[pos+1].shape[1],-1) # update the next core

#         if R.shape[0] != x.rank[pos+1]:
#             x.rank = list(x.rank) # convert the tuple to a list
#             x.rank[pos+1] = R.shape[0] # assign the new value
#             x.rank = tuple(x.rank) # convert the list back to a tuple
#     elif dir.lower() == 'right':
#         # mind the transpose as we want to orthonormalize rows
#         Q, R = tl.qr(Un.reshape(x.rank[pos], -1).T, mode='reduced') # perform QR decomposition on the transpose
#         # Fixed signs of x.U{pos},  if it is orthogonal.
#         # This needs for the ASCU algorithm when it updates ranks both
#         # sides.
#         sR = np.sign(np.diag(R)) # get the signs of the diagonal elements of R
#         Q = Q * sR # multiply Q by the signs
#         R = (R.T * sR).T # multiply R by the signs

#         Un = Q.T.reshape(-1, x.shape[pos], x.rank[pos+1]) # reshape Q transpose to the core shape
#         Un2 = x[pos-1].reshape(-1, x.rank[pos]) @  R.T # multiply the previous core by R transpose

#         x[pos] = Un.reshape(Un.shape[0],x[pos].shape[1],-1) # update the current core
#         x[pos-1] = Un2.reshape(x[pos-1].shape[0],x[pos-1].shape[1],-1) # update the previous core

#         if R.shape[0] != x.rank[pos]:
#             x.rank = list(x.rank) # convert the tuple to a list
#             x.rank[pos] = R.shape[0] # assign the new value
#             x.rank = tuple(x.rank) # convert the list back to a tuple
#     else:
#         raise ValueError('Unknown direction specified. Choose either LEFT or RIGHT')

#     return x


# def tt_orthogonalize(x, pos):
#     """Orthogonalize tensor.

#     x = orthogonalize(x, pos) orthogonalizes all cores of the TTeMPS tensor x
#     except the core at position pos. Cores 1...pos-1 are left-, cores pos+1...end
#     are right-orthogonalized. Therefore,

#     x = orthogonalize(x, 1) right-orthogonalizes the full tensor,

#     x = orthogonalize(x, x.order) left-orthogonalizes the full tensor.

#     See also orth_at.
#     """

#     # adapted from the TTeMPS Toolbox.

#     # left orthogonalization till pos (from left)
#     for i in range(pos):
#         # print(f'Left orthogonalization {i}')
#         x = tt_orth_at(x, i, 'left')

#     # right orthogonalization till pos (from right)
#     ndimX = len(x.factors)

#     for i in range(ndimX-1, pos, -1):
#         # print(f'Right orthogonalization {i}')
#         x = tt_orth_at(x, i, 'right')

#     return x


# def is_left_orthogonal(core):
#     r_prev, n, r_curr = core.shape
#     Gmat = core.reshape(r_prev * n, r_curr)
#     return np.allclose(Gmat.T @ Gmat, np.eye(r_curr))


# def is_right_orthogonal(core):
#     r_prev, n, r_curr = core.shape
#     Gmat = core.reshape(r_prev, n * r_curr)
#     return np.allclose(Gmat @ Gmat.T, np.eye(r_prev))


# def DMD(X1, X2, r, dt):
#         U, s, Vh = np.linalg.svd(X1, full_matrices=False)
#         Ur = U[:, :r]
#         Sr = np.diag(s[:r])
#         Vr = Vh.conj().T[:, :r]

#         Atilde = Ur.conj().T @ X2 @ Vr @ np.linalg.inv(Sr)
#         Lambda, W = np.linalg.eig(Atilde)

#         Phi = X2 @ Vr @ np.linalg.inv(Sr) @ W
#         omega = np.log(Lambda) / dt

#         alpha1 = np.linalg.lstsq(Phi, X1[:, 0], rcond=None)[0]
#         b = np.linalg.lstsq(Phi, X2[:, 0], rcond=None)[0]

#         time_dynamics = None
#         for i in range(X1.shape[1]):
#             v = np.array(alpha1)[:, 0] * np.exp(np.array(omega) * (i + 1) * dt)
#             if time_dynamics is None:
#                 time_dynamics = v
#             else:
#                 time_dynamics = np.vstack((time_dynamics, v))
#         X_dmd = np.dot(np.array(Phi), time_dynamics.T)

#         return Phi, omega, Lambda, alpha1, b, X_dmd, time_dynamics.T


# def matricize_tt(tt_cores, row_modes, col_modes, dims):
#     """
#     Matricizes a Tensor Train (TT) tensor into a matrix.

#     Args:
#         tt_cores (list of numpy.ndarray): List of TT-cores.
#         row_modes (tuple of int): Indices of modes to form rows.
#         col_modes (tuple of int): Indices of modes to form columns.
#         dims (tuple of int): Original dimensions of the tensor.

#     Returns:
#         numpy.ndarray: Matricized tensor.
#     """
#     num_cores = len(tt_cores)
#     row_size = np.prod([dims[i] for i in row_modes])
#     col_size = np.prod([dims[i] for i in col_modes])

#     # Initial contraction
#     core_idx = 0
#     if core_idx in row_modes:
#         row_pos = row_modes.index(core_idx)
#         curr_rows = dims[core_idx]
#     else:
#         row_pos = -1
#         curr_rows = 1

#     if core_idx in col_modes:
#       col_pos = col_modes.index(core_idx)
#       curr_cols = dims[core_idx]
#     else:
#       col_pos = -1
#       curr_cols = 1

#     matrix = tt_cores[core_idx]

#     # Subsequent contractions
#     for core_idx in range(1, num_cores):
#         if core_idx in row_modes:
#           row_pos = row_modes.index(core_idx)
#           next_rows = dims[core_idx]
#         else:
#           row_pos = -1
#           next_rows = 1

#         if core_idx in col_modes:
#           col_pos = col_modes.index(core_idx)
#           next_cols = dims[core_idx]
#         else:
#           col_pos = -1
#           next_cols = 1

#         if row_pos != -1 and col_pos != -1:
#           matrix = np.einsum('aij,jbk->abik', matrix, tt_cores[core_idx])
#           matrix = matrix.reshape(curr_rows*next_rows, curr_cols*next_cols, matrix.shape[-1])
#           curr_rows *= next_rows
#           curr_cols *= next_cols
#         elif row_pos != -1:
#           matrix = np.einsum('aij,jbk->abik', matrix, tt_cores[core_idx])
#           matrix = matrix.reshape(curr_rows*next_rows, matrix.shape[-1])
#           curr_rows *= next_rows
#         elif col_pos != -1:
#           matrix = np.einsum('ai,ijk->ajk', matrix, tt_cores[core_idx])
#           matrix = matrix.reshape(matrix.shape[0], curr_cols*next_cols)
#           curr_cols *= next_cols
#         else:
#           matrix = np.einsum('ai,ijk->ajk', matrix, tt_cores[core_idx])

#     return matrix.reshape(row_size, col_size)


# def pseudoinverse(tt_cores: list, l: int):
#     """Calculate pseudoinverse.

#     Parameters
#     ----------
#         tt_cores : list
#             List of TT cores, each is a np.ndarray with a shape of (r_prev,
#             d_mode, r_next).
#         l : int
#             Core number to split the tensor train and assemble a matrix.

#     Returns
#     -------
#         Pseudoinverse matrix of tt decomposition relative to given core number.
#     """

#     # Step 1. Given a tensor x in TT-format and core number 1 <= l <= d − 1 to
#     # compute pseudoinverse of x(n_1, ..., n_l; n_(l+1), ..., n_d).

#     tt_cores_copy = np.copy(tt_cores)
#     d = tt_cores_copy.shape[0]

#     if not 1 <= l <= d-1:
#         raise ValueError("""Core number must be between 1 and d-1. It is not
#                             possible to split a tensor train otherwise.""")

#     l -= 1

#     # Step 2. Left-orthonormalize x_1, ..., x_(l-1) and right-orthonormalize
#     # x_d, ..., x_(l+1).

#     tt_cores_copy[0:l-1] = left_orthogonalize(tt_cores_copy[0:l-1])
#     tt_cores_copy[l+1:d:-1] = right_orthogonalize(tt_cores_copy[l+1:d:-1])

#     # Step 3. Compute SVD of x_l(r_(l-1), n_l; r_l).

#     x_l = tt_cores[l]
#     r_l_prev, n_l, r_l = x_l.shape
#     x_l_mat = x_l.reshape(r_l_prev * n_l, r_l)

#     U, S, V_T = np.linalg.svd(x_l_mat, full_matrices=False)

#     # Step 4. Define tensor "y".

#     y = U.reshape(r_l_prev, n_l, U.shape[1])

#     # Step 5. Define tensor "z".

#     x_l_next = tt_cores[l]
#     r_l, n_l_next, r_l_next = x_l_next.shape
#     x_l_next_mat = x_l_next.reshape(r_l, n_l_next * r_l_next)

#     z = V_T @ x_l_next_mat

#     # Step 6.

#     tt_cores_copy[l] = y
#     tt_cores_copy[l+1] = z

#     # Step 7. Compute matrix M.

#     k_l_prev = tt_cores_copy[l-1].shape[1]
#     M = np.tensordot(tt_cores_copy[0:l-1], tt_cores_copy[l,k_l_prev,:,:], axes=([-1], [0]))
#     n_prod_left = 1
#     for i in range(l):
#         n_prod_left *= tt_cores_copy[i].shape[2]
#     M = M.reshape(n_prod_left, r_l)

#     # Step 8. Compute matrix N.

#     k_l_next = tt_cores_copy[l+1].shape[1]
#     N = np.tensordot(tt_cores_copy[l+1,:,:,k_l_next], tt_cores_copy[l+2:], axes=([-1], [0]))
#     n_prod_right = 1
#     for i in range(l+1, d):
#         n_prod_left *= tt_cores_copy[i].shape[2]
#     N = N.reshape(r_l, n_prod_right)

#     # Step 9. Compute pseudo-inverse and return.

#     return N.T @ np.inv(Sigma) @ M.T

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd, pinv, eig
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')
import time 
import numpy as np
import torch
import time
from scipy.linalg import svd, qr, eig, pinv
from scipy.sparse.linalg import eigs
from tensorly.decomposition import tensor_train
from tensorly import backend as T
import matplotlib.pyplot as plt

try:
    import torch
    TORCH_AVAILABLE = True
    print("PyTorch available for GPU acceleration")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, using CPU only")

# Set numpy to highest precision
np.set_printoptions(precision=16)

class DMD:
    """
    Dynamic Mode Decomposition with high precision and GPU support
    """
    
    def __init__(self, svd_rank=None, gpu=False, precision='double'):
        """
        Initialize DMD
        
        Parameters:
        -----------
        svd_rank : int or None
            Rank for SVD truncation
        gpu : bool
            Use GPU acceleration if available
        precision : str
            'single', 'double', or 'highest'
        """
        self.svd_rank = svd_rank
        self.gpu = gpu and TORCH_AVAILABLE
        self.precision = precision
        self.device = None
        self.dtype_np = None
        self.dtype_torch = None
        
        self._setup_precision()
        self._setup_device()
        
        # Results storage
        self.eigenvalues = None
        self.modes = None
        self.amplitudes = None
        self.dynamics = None
        self.reconstructed_data = None
        self.original_data = None
        
        # Statistics
        self.stats = {}
        
    def _setup_precision(self):
        """Setup precision types"""
        if self.precision == 'single':
            self.dtype_np = np.float32
            self.dtype_torch = torch.float32
        elif self.precision == 'double':
            self.dtype_np = np.float64
            self.dtype_torch = torch.float64
        else:  # highest
            self.dtype_np = np.longdouble
            self.dtype_torch = torch.float64  # PyTorch doesn't support long double
            
    def _setup_device(self):
        """Setup computation device"""
        if self.gpu and TORCH_AVAILABLE:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = torch.device('cpu')
                self.gpu = False
                print("CUDA not available, falling back to CPU")
        else:
            self.device = torch.device('cpu') if TORCH_AVAILABLE else None
            
    def _to_tensor(self, array):
        """Convert numpy array to torch tensor with proper device and dtype"""
        if self.gpu and TORCH_AVAILABLE:
            return torch.from_numpy(array.astype(self.dtype_np)).to(self.device, dtype=self.dtype_torch)
        return array.astype(self.dtype_np)
    
    def _to_numpy(self, tensor):
        """Convert tensor back to numpy"""
        if self.gpu and TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy().astype(self.dtype_np)
        return tensor.astype(self.dtype_np)
    
    def _svd_decomposition(self, matrix):
        """High-precision SVD decomposition"""
        print(f"Performing SVD decomposition with {self.precision} precision...")
        
        if self.gpu and TORCH_AVAILABLE:
            # GPU SVD using PyTorch
            matrix_tensor = self._to_tensor(matrix)
            try:
                U, S, Vt = torch.linalg.svd(matrix_tensor, full_matrices=False)
                U = self._to_numpy(U)
                S = self._to_numpy(S)
                Vt = self._to_numpy(Vt)
            except Exception as e:
                print(f"GPU SVD failed: {e}, falling back to CPU")
                U, S, Vt = svd(matrix.astype(self.dtype_np), full_matrices=False)
        else:
            # CPU SVD using scipy
            U, S, Vt = svd(matrix.astype(self.dtype_np), full_matrices=False)
        
        # Rank truncation
        if self.svd_rank is not None:
            rank = min(self.svd_rank, len(S))
            U = U[:, :rank]
            S = S[:rank]
            Vt = Vt[:rank, :]
            
        print(f"SVD completed. Matrix shape: {matrix.shape}, Reduced rank: {len(S)}")
        return U, S, Vt
    
    def _eigendecomposition(self, matrix):
        """High-precision eigendecomposition"""
        print("Performing eigendecomposition...")
        
        if self.gpu and TORCH_AVAILABLE:
            # GPU eigendecomposition
            matrix_tensor = self._to_tensor(matrix)
            try:
                eigenvalues, eigenvectors = torch.linalg.eig(matrix_tensor)
                eigenvalues = self._to_numpy(eigenvalues)
                eigenvectors = self._to_numpy(eigenvectors)
            except Exception as e:
                print(f"GPU eigendecomposition failed: {e}, falling back to CPU")
                eigenvalues, eigenvectors = eig(matrix.astype(self.dtype_np))
        else:
            # CPU eigendecomposition
            eigenvalues, eigenvectors = eig(matrix.astype(self.dtype_np))
            
        return eigenvalues, eigenvectors
    
    def fit(self, X):
        """
        Fit DMD to data matrix X
        
        Parameters:
        -----------
        X : np.ndarray
            Data matrix of shape (n_features, n_samples)
        """
        print("="*60)
        print("Starting DMD Analysis")
        print("="*60)
        
        self.original_data = X.astype(self.dtype_np)
        n_features, n_samples = X.shape
        
        print(f"Data shape: {X.shape}")
        print(f"Precision: {self.precision}")
        print(f"Device: {'GPU' if self.gpu else 'CPU'}")
        
        # Split data into X1 and X2
        X1 = X[:, :-1].astype(self.dtype_np)  # snapshots 0 to n-2
        X2 = X[:, 1:].astype(self.dtype_np)   # snapshots 1 to n-1
        
        print(f"X1 shape: {X1.shape}, X2 shape: {X2.shape}")
        
        # Step 1: SVD of X1
        U, S, Vt = self._svd_decomposition(X1)
        
        # Step 2: Compute reduced-order linear map
        # A_tilde = U^T * X2 * V * S^{-1}
        S_inv = np.zeros_like(S, dtype=self.dtype_np)
        S_inv[S > np.finfo(self.dtype_np).eps * 10] = 1.0 / S[S > np.finfo(self.dtype_np).eps * 10]
        
        if self.gpu and TORCH_AVAILABLE:
            U_tensor = self._to_tensor(U)
            X2_tensor = self._to_tensor(X2)
            Vt_tensor = self._to_tensor(Vt)
            S_inv_tensor = self._to_tensor(S_inv)
            
            A_tilde_tensor = torch.mm(torch.mm(U_tensor.T, X2_tensor), 
                                    torch.mm(Vt_tensor.T, torch.diag(S_inv_tensor)))
            A_tilde = self._to_numpy(A_tilde_tensor)
        else:
            A_tilde = U.T @ X2 @ Vt.T @ np.diag(S_inv)
        
        print(f"A_tilde shape: {A_tilde.shape}")
        
        # Step 3: Eigendecomposition of A_tilde
        eigenvalues, W = self._eigendecomposition(A_tilde)
        
        # Step 4: Compute DMD modes
        self.modes = X2 @ Vt.T @ np.diag(S_inv) @ W
        self.eigenvalues = eigenvalues
        
        # Step 5: Compute amplitudes (initial condition)
        self.amplitudes = pinv(self.modes) @ X[:, 0]
        
        print(f"Number of modes: {len(self.eigenvalues)}")
        print(f"Eigenvalue range: [{np.min(np.abs(eigenvalues)):.2e}, {np.max(np.abs(eigenvalues)):.2e}]")
        
        return self
    
    def predict(self, time_steps=None):
        """
        Predict/reconstruct data using DMD
        
        Parameters:
        -----------
        time_steps : int or None
            Number of time steps to predict
        """
        if self.modes is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        n_samples = self.original_data.shape[1] if time_steps is None else time_steps
        
        # Time dynamics
        time_dynamics = np.zeros((len(self.eigenvalues), n_samples), dtype=complex)
        for i, lam in enumerate(self.eigenvalues):
            time_dynamics[i, :] = [lam**t for t in range(n_samples)]
        
        # Reconstruct
        self.dynamics = time_dynamics
        self.reconstructed_data = np.real(self.modes @ np.diag(self.amplitudes) @ time_dynamics)
        
        return self.reconstructed_data
    
    def compute_statistics(self):
        """Compute comprehensive reconstruction statistics"""
        if self.reconstructed_data is None:
            self.predict()
        
        original = self.original_data
        reconstructed = self.reconstructed_data
        
        # Ensure same shape
        min_samples = min(original.shape[1], reconstructed.shape[1])
        original = original[:, :min_samples]
        reconstructed = reconstructed[:, :min_samples]
        
        # Basic error metrics
        mse = np.mean((original - reconstructed) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(original - reconstructed))
        
        # Normalized errors
        frobenius_error = np.linalg.norm(original - reconstructed, 'fro')
        frobenius_norm = np.linalg.norm(original, 'fro')
        relative_error = frobenius_error / frobenius_norm
        
        # Energy metrics
        original_energy = np.sum(original ** 2)
        reconstructed_energy = np.sum(reconstructed ** 2)
        energy_ratio = reconstructed_energy / original_energy
        
        # Correlation
        orig_flat = original.flatten()
        recon_flat = reconstructed.flatten()
        correlation, _ = pearsonr(orig_flat, recon_flat)
        
        # R-squared
        ss_res = np.sum((orig_flat - recon_flat) ** 2)
        ss_tot = np.sum((orig_flat - np.mean(orig_flat)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Signal-to-noise ratio
        signal_power = np.mean(original ** 2)
        noise_power = np.mean((original - reconstructed) ** 2)
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf
        
        # Eigenvalue analysis
        stable_modes = np.sum(np.abs(self.eigenvalues) <= 1.0)
        unstable_modes = np.sum(np.abs(self.eigenvalues) > 1.0)
        
        self.stats = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'relative_error': relative_error,
            'energy_ratio': energy_ratio,
            'correlation': correlation,
            'r_squared': r_squared,
            'snr_db': snr_db,
            'stable_modes': stable_modes,
            'unstable_modes': unstable_modes,
            'total_modes': len(self.eigenvalues),
            'condition_number': np.linalg.cond(self.modes) if self.modes is not None else np.inf
        }
        
        return self.stats
    
    def visualize_results(self, figsize=(20, 20)):
        """Comprehensive visualization of HODMD/DMD results including error, eigenvalues, amplitudes, and statistics."""
        if self.reconstructed_data is None:
            self.predict()
        
        fig = plt.figure(figsize=figsize)

        # Original and reconstructed data
        ax1 = plt.subplot(3, 3, 1)
        plt.contourf(np.real(np.reshape(self.original_data[:,0],(449,199))).T, levels = 1001, vmin=-2, vmax=2)
        plt.scatter(50,100,900,color='white', zorder=2) # draw cylinder
        plt.title('Original Data')
        plt.ylabel('Spatial Mode')
        plt.xlabel('Time')

        ax2 = plt.subplot(3, 3, 2)
        plt.contourf(np.real(np.reshape(self.reconstructed_data[:,0],(449,199))).T, levels = 1001, vmin=-2, vmax=2)
        plt.scatter(50,100,900,color='white', zorder=2) # draw cylinder
        plt.title('Reconstructed Data by DMD matrix')
        plt.ylabel('Spatial Mode')
        plt.xlabel('Time')

        ax3 = plt.subplot(3, 3, 3)
        error_data = self.original_data - self.reconstructed_data
        im3 = plt.imshow(error_data, aspect='auto', cmap='RdBu_r')
        plt.title('Reconstruction Error')
        plt.ylabel('Spatial Mode')
        plt.colorbar(im3)

        # Eigenvalue plot
        ax4 = plt.subplot(3, 3, 4)
        plt.scatter(np.real(self.eigenvalues), np.imag(self.eigenvalues),
                    c=np.abs(self.eigenvalues), cmap='plasma', s=60, alpha=0.7)
        plt.colorbar(label='|λ|')
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='Unit Circle')
        plt.xlabel('Real(λ)')
        plt.ylabel('Imag(λ)')
        plt.title('Eigenvalues')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')

        # Mode amplitudes
        ax5 = plt.subplot(3, 3, 5)
        plt.semilogy(np.abs(self.amplitudes), 'bo-', markersize=4)
        plt.xlabel('Mode Index')
        plt.ylabel('|Amplitude|')
        plt.title('Mode Amplitudes')
        plt.grid(True, alpha=0.3)

        # Time series comparison
        ax6 = plt.subplot(3, 3, 6)
        mid_spatial = self.original_data.shape[0] // 2
        plt.plot(self.original_data[mid_spatial, :], 'b-', label='Original', alpha=0.7)
        plt.plot(self.reconstructed_data[mid_spatial, :], 'r--', label='Reconstructed', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title(f'Time Series (Spatial Mode {mid_spatial})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Rank vs Accuracy (MSE)
        ax7 = plt.subplot(3, 3, 7)
        if hasattr(self, 'benchmark_data') and 'rank' in self.benchmark_data:
            ranks = self.benchmark_data['rank']
            mses = self.benchmark_data['rmse']
            plt.plot(ranks, mses, 'go-', lw=2)
            plt.xlabel('SVD Rank')
            plt.ylabel('RMSE')
            plt.title('Rank vs Accuracy (RMSE)')
            plt.grid(True, alpha=0.3)
            plt.yscale("log")

        # Speed vs Accuracy (GPU only)
        ax8 = plt.subplot(3, 3, 8)
        if hasattr(self, 'benchmark_data') and 'time' in self.benchmark_data:
            times = self.benchmark_data['time']
            mses = self.benchmark_data['rmse']
            plt.plot(times, mses, 'mo-', lw=2)
            plt.xlabel('Time (s)')
            plt.ylabel('MSE')
            plt.title('Speed vs Accuracy')
            plt.grid(True, alpha=0.3)
            plt.yscale("log")

        # Rank vs Speed
        ax9 = plt.subplot(3, 3, 9)
        if hasattr(self, 'benchmark_data') and 'time' in self.benchmark_data:
            ranks = self.benchmark_data['rank']
            times = self.benchmark_data['time']
            plt.plot(ranks, times, 'co-', lw=2)
            plt.xlabel('SVD Rank')
            plt.ylabel('Time (s)')
            plt.title('Rank vs Computation Time')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    def benchmark_analysis(self, X, rank_list=None):
        """
        Benchmark DMD/HODMD at different SVD ranks.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data matrix
        rank_list : list
            List of SVD ranks to test
        
        Returns:
        --------
        dict with keys: 'rank', 'mse', 'rmse', 'time'
        """
        if rank_list is None:
            rank_list = [1, 2, 5, 10, 20, 50]
        
        benchmark_data = {
            'rank': [],
            'mse': [],
            'rmse': [],
            'time': []
        }

        print("Starting benchmark analysis...")
        for rank in rank_list:
            print(f"Testing SVD rank: {rank}")
            self.svd_rank = rank
            start_time = time.time()
            self.fit(X)
            self.predict()
            stats = self.compute_statistics()
            elapsed = time.time() - start_time
            
            benchmark_data['rank'].append(rank)
            benchmark_data['mse'].append(stats['mse'])
            benchmark_data['rmse'].append(stats['rmse'])
            benchmark_data['time'].append(elapsed)

        self.benchmark_data = benchmark_data
        print("Benchmark completed.")
        return benchmark_data
    


class HODMD:
    """
    Higher-Order Dynamic Mode Decomposition with high precision and GPU support
    """
    
    def __init__(self, d=1, svd_rank=None, gpu=False, precision='double'):
        """
        Initialize HODMD
        
        Parameters:
        -----------
        d : int
            Time delay embedding dimension
        svd_rank : int or None
            Rank for SVD truncation
        gpu : bool
            Use GPU acceleration if available
        precision : str
            'single', 'double', or 'highest'
        """
        self.d = d
        self.svd_rank = svd_rank
        self.gpu = gpu and TORCH_AVAILABLE
        self.precision = precision
        self.device = None
        self.dtype_np = None
        self.dtype_torch = None
        
        self._setup_precision()
        self._setup_device()
        
        # Results storage
        self.eigenvalues = None
        self.modes = None
        self.amplitudes = None
        self.dynamics = None
        self.reconstructed_data = None
        self.original_data = None
        self.hankel_matrix = None
        
        # Statistics
        self.stats = {}
        
    def _setup_precision(self):
        """Setup precision types"""
        if self.precision == 'single':
            self.dtype_np = np.float32
            self.dtype_torch = torch.float32
        elif self.precision == 'double':
            self.dtype_np = np.float64
            self.dtype_torch = torch.float64
        else:  # highest
            self.dtype_np = np.longdouble
            self.dtype_torch = torch.float64
            
    def _setup_device(self):
        """Setup computation device"""
        if self.gpu and TORCH_AVAILABLE:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = torch.device('cpu')
                self.gpu = False
                print("CUDA not available, falling back to CPU")
        else:
            self.device = torch.device('cpu') if TORCH_AVAILABLE else None
            
    def _to_tensor(self, array):
        """Convert numpy array to torch tensor"""
        if self.gpu and TORCH_AVAILABLE:
            return torch.from_numpy(array.astype(self.dtype_np)).to(self.device, dtype=self.dtype_torch)
        return array.astype(self.dtype_np)
    
    def _to_numpy(self, tensor):
        """Convert tensor back to numpy"""
        if self.gpu and TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy().astype(self.dtype_np)
        return tensor.astype(self.dtype_np)
    
    def _create_hankel_matrix(self, X):
        """Create Hankel matrix for time-delay embedding"""
        print(f"Creating Hankel matrix with delay d={self.d}...")
        
        n_features, n_samples = X.shape
        
        if self.d == 1:
            return X.astype(self.dtype_np)
        
        # Create Hankel matrix
        n_cols = n_samples - self.d + 1
        hankel = np.zeros((n_features * self.d, n_cols), dtype=self.dtype_np)
        
        for i in range(self.d):
            hankel[i*n_features:(i+1)*n_features, :] = X[:, i:i+n_cols]
        
        print(f"Hankel matrix shape: {hankel.shape}")
        return hankel
    
    def _svd_decomposition(self, matrix):
        """High-precision SVD decomposition"""
        print(f"Performing SVD decomposition with {self.precision} precision...")
        
        if self.gpu and TORCH_AVAILABLE:
            matrix_tensor = self._to_tensor(matrix)
            try:
                U, S, Vt = torch.linalg.svd(matrix_tensor, full_matrices=False)
                U = self._to_numpy(U)
                S = self._to_numpy(S)
                Vt = self._to_numpy(Vt)
            except Exception as e:
                print(f"GPU SVD failed: {e}, falling back to CPU")
                U, S, Vt = svd(matrix.astype(self.dtype_np), full_matrices=False)
        else:
            U, S, Vt = svd(matrix.astype(self.dtype_np), full_matrices=False)
        
        # Rank truncation
        if self.svd_rank is not None:
            rank = min(self.svd_rank, len(S))
            U = U[:, :rank]
            S = S[:rank]
            Vt = Vt[:rank, :]
            
        print(f"SVD completed. Reduced rank: {len(S)}")
        return U, S, Vt
    
    def _eigendecomposition(self, matrix):
        """High-precision eigendecomposition"""
        print("Performing eigendecomposition...")
        
        if self.gpu and TORCH_AVAILABLE:
            matrix_tensor = self._to_tensor(matrix)
            try:
                eigenvalues, eigenvectors = torch.linalg.eig(matrix_tensor)
                eigenvalues = self._to_numpy(eigenvalues)
                eigenvectors = self._to_numpy(eigenvectors)
            except Exception as e:
                print(f"GPU eigendecomposition failed: {e}, falling back to CPU")
                eigenvalues, eigenvectors = eig(matrix.astype(self.dtype_np))
        else:
            eigenvalues, eigenvectors = eig(matrix.astype(self.dtype_np))
            
        return eigenvalues, eigenvectors
    
    def fit(self, X):
        """
        Fit HODMD to data matrix X
        
        Parameters:
        -----------
        X : np.ndarray
            Data matrix of shape (n_features, n_samples)
        """
        print("="*60)
        print("Starting HODMD Analysis")
        print("="*60)
        
        self.original_data = X.astype(self.dtype_np)
        
        print(f"Data shape: {X.shape}")
        print(f"Time delay: {self.d}")
        print(f"Precision: {self.precision}")
        print(f"Device: {'GPU' if self.gpu else 'CPU'}")
        
        # Step 1: Create Hankel matrix
        self.hankel_matrix = self._create_hankel_matrix(X)
        
        # Step 2: Split Hankel matrix
        H1 = self.hankel_matrix[:, :-1]  # First n-1 columns
        H2 = self.hankel_matrix[:, 1:]   # Last n-1 columns
        
        print(f"H1 shape: {H1.shape}, H2 shape: {H2.shape}")
        
        # Step 3: SVD of H1
        U, S, Vt = self._svd_decomposition(H1)
        
        # Step 4: Compute reduced-order operator
        S_inv = np.zeros_like(S, dtype=self.dtype_np)
        S_inv[S > np.finfo(self.dtype_np).eps * 10] = 1.0 / S[S > np.finfo(self.dtype_np).eps * 10]
        
        if self.gpu and TORCH_AVAILABLE:
            U_tensor = self._to_tensor(U)
            H2_tensor = self._to_tensor(H2)
            Vt_tensor = self._to_tensor(Vt)
            S_inv_tensor = self._to_tensor(S_inv)
            
            A_tilde_tensor = torch.mm(torch.mm(U_tensor.T, H2_tensor), 
                                    torch.mm(Vt_tensor.T, torch.diag(S_inv_tensor)))
            A_tilde = self._to_numpy(A_tilde_tensor)
        else:
            A_tilde = U.T @ H2 @ Vt.T @ np.diag(S_inv)
        
        # Step 5: Eigendecomposition
        eigenvalues, W = self._eigendecomposition(A_tilde)
        
        # Step 6: Compute HODMD modes
        self.modes = H2 @ Vt.T @ np.diag(S_inv) @ W
        self.eigenvalues = eigenvalues
        
        # Step 7: Compute amplitudes
        self.amplitudes = pinv(self.modes) @ self.hankel_matrix[:, 0]
        
        print(f"Number of modes: {len(self.eigenvalues)}")
        print(f"Eigenvalue range: [{np.min(np.abs(eigenvalues)):.2e}, {np.max(np.abs(eigenvalues)):.2e}]")
        
        return self
    
    def predict(self, time_steps=None):
        """Predict/reconstruct data using HODMD"""
        if self.modes is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        n_samples = self.hankel_matrix.shape[1] if time_steps is None else time_steps
        
        # Time dynamics
        time_dynamics = np.zeros((len(self.eigenvalues), n_samples), dtype=complex)
        for i, lam in enumerate(self.eigenvalues):
            time_dynamics[i, :] = [lam**t for t in range(n_samples)]
        
        # Reconstruct Hankel matrix
        hankel_reconstructed = np.real(self.modes @ np.diag(self.amplitudes) @ time_dynamics)
        
        # Extract original data from Hankel matrix (anti-diagonal averaging)
        n_features = self.original_data.shape[0]
        n_time_orig = self.original_data.shape[1]
        
        if self.d == 1:
            self.reconstructed_data = hankel_reconstructed
        else:
            # Anti-diagonal averaging to reconstruct original time series
            reconstructed = np.zeros((n_features, n_time_orig), dtype=self.dtype_np)
            
            for t in range(n_time_orig):
                count = 0
                for i in range(self.d):
                    if t - i >= 0 and t - i < hankel_reconstructed.shape[1]:
                        reconstructed[:, t] += hankel_reconstructed[i*n_features:(i+1)*n_features, t-i]
                        count += 1
                if count > 0:
                    reconstructed[:, t] /= count
            
            self.reconstructed_data = reconstructed
        
        self.dynamics = time_dynamics
        return self.reconstructed_data
    
    def compute_statistics(self):
        """Compute comprehensive reconstruction statistics"""
        if self.reconstructed_data is None:
            self.predict()
        
        original = self.original_data
        reconstructed = self.reconstructed_data
        
        # Ensure same shape
        min_samples = min(original.shape[1], reconstructed.shape[1])
        original = original[:, :min_samples]
        reconstructed = reconstructed[:, :min_samples]
        
        # All the same statistics as DMD
        mse = np.mean((original - reconstructed) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(original - reconstructed))
        
        frobenius_error = np.linalg.norm(original - reconstructed, 'fro')
        frobenius_norm = np.linalg.norm(original, 'fro')
        relative_error = frobenius_error / frobenius_norm
        
        original_energy = np.sum(original ** 2)
        reconstructed_energy = np.sum(reconstructed ** 2)
        energy_ratio = reconstructed_energy / original_energy
        
        orig_flat = original.flatten()
        recon_flat = reconstructed.flatten()
        correlation, _ = pearsonr(orig_flat, recon_flat)
        
        ss_res = np.sum((orig_flat - recon_flat) ** 2)
        ss_tot = np.sum((orig_flat - np.mean(orig_flat)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        signal_power = np.mean(original ** 2)
        noise_power = np.mean((original - reconstructed) ** 2)
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf
        
        stable_modes = np.sum(np.abs(self.eigenvalues) <= 1.0)
        unstable_modes = np.sum(np.abs(self.eigenvalues) > 1.0)
        
        self.stats = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'relative_error': relative_error,
            'energy_ratio': energy_ratio,
            'correlation': correlation,
            'r_squared': r_squared,
            'snr_db': snr_db,
            'stable_modes': stable_modes,
            'unstable_modes': unstable_modes,
            'total_modes': len(self.eigenvalues),
            'condition_number': np.linalg.cond(self.modes) if self.modes is not None else np.inf,
            'hankel_rank': np.linalg.matrix_rank(self.hankel_matrix) if self.hankel_matrix is not None else 0
        }
        
        return self.stats
    
    def visualize_results(self, figsize=(20, 20)):
        """Comprehensive visualization of HODMD/DMD results including error, eigenvalues, amplitudes, and statistics."""
        if self.reconstructed_data is None:
            self.predict()
        
        fig = plt.figure(figsize=figsize)

        # Original and reconstructed data
        ax1 = plt.subplot(3, 3, 1)
        plt.contourf(np.real(np.reshape(self.original_data[:,0],(449,199))).T, levels = 1001, vmin=-2, vmax=2)
        plt.scatter(50,100,900,color='white', zorder=2) # draw cylinder
        plt.title('Original Data')
        plt.ylabel('Spatial Mode')
        plt.xlabel('Time')

        ax2 = plt.subplot(3, 3, 2)
        plt.contourf(np.real(np.reshape(self.reconstructed_data[:,0],(449,199))).T, levels = 1001, vmin=-2, vmax=2)
        plt.scatter(50,100,900,color='white', zorder=2) # draw cylinder
        plt.title('Reconstructed Data by DMD matrix')
        plt.ylabel('Spatial Mode')
        plt.xlabel('Time')

        ax3 = plt.subplot(3, 3, 3)
        error_data = self.original_data - self.reconstructed_data
        im3 = plt.imshow(error_data, aspect='auto', cmap='RdBu_r')
        plt.title('Reconstruction Error')
        plt.ylabel('Spatial Mode')
        plt.colorbar(im3)

        # Eigenvalue plot
        ax4 = plt.subplot(3, 3, 4)
        plt.scatter(np.real(self.eigenvalues), np.imag(self.eigenvalues),
                    c=np.abs(self.eigenvalues), cmap='plasma', s=60, alpha=0.7)
        plt.colorbar(label='|λ|')
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='Unit Circle')
        plt.xlabel('Real(λ)')
        plt.ylabel('Imag(λ)')
        plt.title('Eigenvalues')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')

        # Mode amplitudes
        ax5 = plt.subplot(3, 3, 5)
        plt.semilogy(np.abs(self.amplitudes), 'bo-', markersize=4)
        plt.xlabel('Mode Index')
        plt.ylabel('|Amplitude|')
        plt.title('Mode Amplitudes')
        plt.grid(True, alpha=0.3)

        # Time series comparison
        ax6 = plt.subplot(3, 3, 6)
        mid_spatial = self.original_data.shape[0] // 2
        plt.plot(self.original_data[mid_spatial, :], 'b-', label='Original', alpha=0.7)
        plt.plot(self.reconstructed_data[mid_spatial, :], 'r--', label='Reconstructed', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title(f'Time Series (Spatial Mode {mid_spatial})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Rank vs Accuracy (MSE)
        ax7 = plt.subplot(3, 3, 7)
        if hasattr(self, 'benchmark_data') and 'rank' in self.benchmark_data:
            ranks = self.benchmark_data['rank']
            mses = self.benchmark_data['rmse']
            plt.plot(ranks, mses, 'go-', lw=2)
            plt.xlabel('SVD Rank')
            plt.ylabel('RMSE')
            plt.title('Rank vs Accuracy (RMSE)')
            plt.grid(True, alpha=0.3)
            plt.yscale("log")

        # Speed vs Accuracy (GPU only)
        ax8 = plt.subplot(3, 3, 8)
        if hasattr(self, 'benchmark_data') and 'time' in self.benchmark_data:
            times = self.benchmark_data['time']
            mses = self.benchmark_data['rmse']
            plt.plot(times, mses, 'mo-', lw=2)
            plt.xlabel('Time (s)')
            plt.ylabel('MSE')
            plt.title('Speed vs Accuracy')
            plt.grid(True, alpha=0.3)
            plt.yscale("log")

        # Rank vs Speed
        ax9 = plt.subplot(3, 3, 9)
        if hasattr(self, 'benchmark_data') and 'time' in self.benchmark_data:
            ranks = self.benchmark_data['rank']
            times = self.benchmark_data['time']
            plt.plot(ranks, times, 'co-', lw=2)
            plt.xlabel('SVD Rank')
            plt.ylabel('Time (s)')
            plt.title('Rank vs Computation Time')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    def benchmark_analysis(self, X, rank_list=None):
        """
        Benchmark DMD/HODMD at different SVD ranks.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data matrix
        rank_list : list
            List of SVD ranks to test
        
        Returns:
        --------
        dict with keys: 'rank', 'mse', 'rmse', 'time'
        """
        if rank_list is None:
            rank_list = [1, 2, 5, 10, 20, 50]
        
        benchmark_data = {
            'rank': [],
            'mse': [],
            'rmse': [],
            'time': []
        }

        print("Starting benchmark analysis...")
        for rank in rank_list:
            print(f"Testing SVD rank: {rank}")
            self.svd_rank = rank
            start_time = time.time()
            self.fit(X)
            self.predict()
            stats = self.compute_statistics()
            elapsed = time.time() - start_time
            
            benchmark_data['rank'].append(rank)
            benchmark_data['mse'].append(stats['mse'])
            benchmark_data['rmse'].append(stats['rmse'])
            benchmark_data['time'].append(elapsed)

        self.benchmark_data = benchmark_data
        print("Benchmark completed.")
        return benchmark_data
    
    import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, qr, pinv
from scipy.sparse.linalg import eigs
import tensorly as tl
from tensorly.decomposition import tensor_train
import warnings
warnings.filterwarnings('ignore')

class HODMD_TT:
    """
    Higher-Order Dynamic Mode Decomposition using Tensor Train decomposition
    """
    
    def __init__(self, rank=None, svd_rank=None):
        """
        Initialize HODMD with TT decomposition
        
        Parameters:
        -----------
        rank : list or int
            TT ranks for decomposition
        svd_rank : int
            SVD rank for DMD computation
        """
        self.rank = rank
        self.svd_rank = svd_rank
        self.tt_cores = None
        self.dmd_modes = None
        self.eigenvalues = None
        self.reconstruction_error = None
        
    def tensorize_data(self, f):
        """
        Convert 3D data to tensor format
        
        Parameters:
        -----------
        f : np.ndarray
            Input data of shape (nx, ny, nt)
            
        Returns:
        --------
        tensor : np.ndarray
            Tensorized data
        """
        print(f"Input data shape: {f.shape}")
        return f
    
    def algorithm1_tt_decomposition(self, tensor):
        """
        Algorithm 1: Convert tensor into TT-format using low-rank decomposition
        
        Parameters:
        -----------
        tensor : np.ndarray
            Input tensor
            
        Returns:
        --------
        tt_cores : list
            List of TT cores
        """
        print("Algorithm 1: TT Decomposition")
        
        # Set default ranks if not provided
        if self.rank is None:
            # Adaptive rank selection based on tensor dimensions
            dims = tensor.shape
            max_rank = min(50, min(dims)//2)
            self.rank = [1] + [max_rank] * (len(dims)-1) + [1]
        
        # Perform TT decomposition using tensorly
        try:
            tt_tensor = tensor_train(tensor, rank=self.rank)
            self.tt_cores = tt_tensor.factors
            print(f"TT decomposition successful with ranks: {[core.shape for core in self.tt_cores]}")
            return self.tt_cores
        except Exception as e:
            print(f"TensorLy decomposition failed, using manual SVD-based approach: {e}")
            return self._manual_tt_decomposition(tensor)
    
    def _manual_tt_decomposition(self, tensor):
        """
        Manual TT decomposition using SVD
        """
        cores = []
        current_tensor = tensor.copy()
        
        for i in range(len(tensor.shape) - 1):
            # Reshape for SVD
            shape = current_tensor.shape
            matrix = current_tensor.reshape(shape[0], -1)
            
            # SVD
            U, s, Vt = svd(matrix, full_matrices=False)
            
            # Determine rank
            if isinstance(self.rank, list):
                r = min(self.rank[i+1], len(s))
            else:
                r = min(self.rank or 10, len(s))
            
            # Truncate
            U = U[:, :r]
            s = s[:r]
            Vt = Vt[:r, :]
            
            # Store core
            if i == 0:
                cores.append(U.reshape(1, shape[0], r))
            else:
                cores.append(U.reshape(U.shape[0], shape[0]//U.shape[0], r))
            
            # Update tensor for next iteration
            current_tensor = (np.diag(s) @ Vt).reshape((r,) + shape[1:])
        
        # Last core
        cores.append(current_tensor.reshape(current_tensor.shape + (1,)))
        
        self.tt_cores = cores
        print(f"Manual TT decomposition completed with {len(cores)} cores")
        return cores
    
    def algorithm2_orthogonalization(self, tt_cores):
        """
        Algorithm 2: Left and right orthogonalization of TT cores
        
        Parameters:
        -----------
        tt_cores : list
            List of TT cores
            
        Returns:
        --------
        left_ortho_cores : list
            Left-orthogonalized cores
        right_ortho_cores : list
            Right-orthogonalized cores
        """
        print("Algorithm 2: Orthogonalization")
        
        n_cores = len(tt_cores)
        left_ortho_cores = [None] * n_cores
        right_ortho_cores = [None] * n_cores
        
        # Left orthogonalization
        for i in range(n_cores - 1):
            core = tt_cores[i]
            r_left, n_i, r_right = core.shape
            
            # Reshape for QR decomposition
            matrix = core.reshape(r_left * n_i, r_right)
            Q, R = qr(matrix, mode='economic')
            
            # Update current core
            left_ortho_cores[i] = Q.reshape(r_left, n_i, Q.shape[1])
            
            # Update next core
            if i < n_cores - 1:
                next_core = tt_cores[i + 1]
                r_left_next, n_next, r_right_next = next_core.shape
                next_matrix = next_core.reshape(r_left_next, n_next * r_right_next)
                updated_next = R @ next_matrix
                tt_cores[i + 1] = updated_next.reshape(R.shape[0], n_next, r_right_next)
        
        left_ortho_cores[-1] = tt_cores[-1]
        
        # Right orthogonalization
        tt_cores_copy = [core.copy() for core in left_ortho_cores]
        
        for i in range(n_cores - 1, 0, -1):
            core = tt_cores_copy[i]
            r_left, n_i, r_right = core.shape
            
            # Reshape for QR decomposition
            matrix = core.reshape(r_left, n_i * r_right)
            Q, R = qr(matrix.T, mode='economic')
            Q = Q.T
            R = R.T
            
            # Update current core
            right_ortho_cores[i] = Q.reshape(Q.shape[0], n_i, r_right)
            
            # Update previous core
            if i > 0:
                prev_core = tt_cores_copy[i - 1]
                r_left_prev, n_prev, r_right_prev = prev_core.shape
                prev_matrix = prev_core.reshape(r_left_prev * n_prev, r_right_prev)
                updated_prev = prev_matrix @ R
                tt_cores_copy[i - 1] = updated_prev.reshape(r_left_prev, n_prev, R.shape[1])
        
        right_ortho_cores[0] = tt_cores_copy[0]
        
        print("Orthogonalization completed")
        return left_ortho_cores, right_ortho_cores
    
    def algorithm3_pseudo_inverse(self, left_cores, right_cores):
        """
        Algorithm 3: Pseudo inverse of matricization of TT-cores tensor
        
        Parameters:
        -----------
        left_cores : list
            Left-orthogonalized cores
        right_cores : list
            Right-orthogonalized cores
            
        Returns:
        --------
        pseudo_inv : np.ndarray
            Pseudo inverse matrix
        """
        print("Algorithm 3: Pseudo Inverse Computation")
        
        n_cores = len(left_cores)
        middle_idx = n_cores // 2
        
        # SVD on middle core
        middle_core = left_cores[middle_idx]
        r_left, n_i, r_right = middle_core.shape
        matrix = middle_core.reshape(r_left, n_i * r_right)
        
        U, s, Vt = svd(matrix, full_matrices=False)
        
        # Truncate based on svd_rank
        if self.svd_rank:
            rank = min(self.svd_rank, len(s))
            U = U[:, :rank]
            s = s[:rank]
            Vt = Vt[:rank, :]
        
        # Compute pseudo inverse
        s_inv = np.zeros_like(s)
        s_inv[s > 1e-12] = 1.0 / s[s > 1e-12]
        pseudo_inv = Vt.T @ np.diag(s_inv) @ U.T
        
        print(f"Pseudo inverse computed with effective rank: {np.sum(s > 1e-12)}")
        return pseudo_inv, U, s, Vt
    
    def algorithm4_dmd_computation(self, tensor, pseudo_inv_data):
        """
        Algorithm 4: Computation of DMD modes and eigenvalues in TT-format
        
        Parameters:
        -----------
        tensor : np.ndarray
            Original tensor
        pseudo_inv_data : tuple
            Pseudo inverse computation results
            
        Returns:
        --------
        eigenvalues : np.ndarray
            DMD eigenvalues
        modes : np.ndarray
            DMD modes
        """
        print("Algorithm 4: DMD Computation")
        
        pseudo_inv, U, s, Vt = pseudo_inv_data
        
        # Create data matrices for DMD
        nx, ny, nt = tensor.shape
        
        # Reshape tensor to matrix form for DMD
        X = tensor.reshape(nx * ny, nt)
        X1 = X[:, :-1]  # Data matrix
        X2 = X[:, 1:]   # Shifted data matrix
        
        # DMD computation
        # Compute A_tilde = U^T * X2 * X1^T * U * S^{-1}
        try:
            # SVD of X1 for better numerical stability
            U_x, s_x, Vt_x = svd(X1, full_matrices=False)
            
            # Truncate based on energy or rank
            energy_threshold = 0.99
            cumulative_energy = np.cumsum(s_x**2) / np.sum(s_x**2)
            effective_rank = np.argmax(cumulative_energy >= energy_threshold) + 1
            effective_rank = min(effective_rank, len(s_x))
            
            U_x = U_x[:, :effective_rank]
            s_x = s_x[:effective_rank]
            Vt_x = Vt_x[:effective_rank, :]
            
            # Compute reduced-order DMD
            A_tilde = U_x.T @ X2 @ Vt_x.T @ np.diag(1.0/s_x)
            
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eig(A_tilde)
            
            # Compute DMD modes
            modes = X2 @ Vt_x.T @ np.diag(1.0/s_x) @ eigenvectors
            
            print(f"DMD computation completed with {len(eigenvalues)} modes")
            print(f"Eigenvalue magnitudes range: [{np.min(np.abs(eigenvalues)):.6f}, {np.max(np.abs(eigenvalues)):.6f}]")
            
        except Exception as e:
            print(f"Advanced DMD failed, using simple approach: {e}")
            # Fallback to simple DMD
            A_approx = X2 @ pinv(X1)
            eigenvalues, modes = eigs(A_approx, k=min(10, min(X1.shape)-1))
        
        self.eigenvalues = eigenvalues
        self.dmd_modes = modes
        
        return eigenvalues, modes
    
    def algorithm5_reconstruction(self, original_tensor, modes, eigenvalues, dt=1.0):
        """
        Algorithm 5: Reconstruction of data and visualization
        
        Parameters:
        -----------
        original_tensor : np.ndarray
            Original tensor data
        modes : np.ndarray
            DMD modes
        eigenvalues : np.ndarray
            DMD eigenvalues
        dt : float
            Time step
            
        Returns:
        --------
        reconstructed_tensor : np.ndarray
            Reconstructed tensor
        """
        print("Algorithm 5: Data Reconstruction")
        
        nx, ny, nt = original_tensor.shape
        
        # Time vector
        t = np.arange(nt) * dt
        
        # Compute time dynamics
        time_dynamics = np.zeros((len(eigenvalues), nt), dtype=complex)
        for i, lam in enumerate(eigenvalues):
            time_dynamics[i, :] = np.power(lam, t)
        
        # Initial conditions (project first snapshot onto modes)
        X_flat = original_tensor.reshape(nx * ny, nt)
        b = pinv(modes) @ X_flat[:, 0]
        
        # Reconstruct
        X_dmd = modes @ np.diag(b) @ time_dynamics
        
        # Take real part and reshape
        X_reconstructed = np.real(X_dmd).reshape(nx, ny, nt)
        
        # Compute reconstruction error
        self.reconstruction_error = np.linalg.norm(original_tensor - X_reconstructed) / np.linalg.norm(original_tensor)
        
        print(f"Reconstruction completed with relative error: {self.reconstruction_error:.6f}")
        
        return X_reconstructed
    
    def algorithm6_statistics(self, original_tensor, reconstructed_tensor):
        """
        Algorithm 6: Statistics to assess reconstruction quality
        
        Parameters:
        -----------
        original_tensor : np.ndarray
            Original tensor
        reconstructed_tensor : np.ndarray
            Reconstructed tensor
            
        Returns:
        --------
        stats : dict
            Dictionary containing various quality metrics
        """
        print("Algorithm 6: Quality Assessment")
        
        # Various error metrics
        print(f"Shape of original tensor: {original_tensor.shape}")
        print(f"Shape of reconstructed tensor: {reconstructed_tensor.shape}")
        frobenius_error = np.linalg.norm(original_tensor - reconstructed_tensor)
        relative_error = frobenius_error / np.linalg.norm(original_tensor)
        
        # Energy preservation
        original_energy = np.sum(original_tensor**2)
        reconstructed_energy = np.sum(reconstructed_tensor**2)
        energy_ratio = reconstructed_energy / original_energy
        
        # Correlation coefficient
        orig_flat = original_tensor.flatten()
        recon_flat = reconstructed_tensor.flatten()
        correlation = np.corrcoef(orig_flat, recon_flat)[0, 1]
        
        # Eigenvalue analysis
        stable_modes = np.sum(np.abs(self.eigenvalues) <= 1.0)
        unstable_modes = np.sum(np.abs(self.eigenvalues) > 1.0)
        
        stats = {
            'frobenius_error': frobenius_error,
            'relative_error': relative_error,
            'energy_ratio': energy_ratio,
            'correlation': correlation,
            'stable_modes': stable_modes,
            'unstable_modes': unstable_modes,
            'total_modes': len(self.eigenvalues),
            'effective_rank': np.sum(np.abs(self.eigenvalues) > 1e-10)
        }
        
        print(f"Quality Statistics:")
        print(f"  Relative Error: {relative_error:.6f}")
        print(f"  Energy Ratio: {energy_ratio:.6f}")
        print(f"  Correlation: {correlation:.6f}")
        print(f"  Stable/Unstable Modes: {stable_modes}/{unstable_modes}")
        
        return stats
    
    def fit(self, f):
        """
        Complete HODMD-TT pipeline
        
        Parameters:
        -----------
        f : np.ndarray
            Input tensor of shape (nx, ny, nt)
            
        Returns:
        --------
        reconstructed : np.ndarray
            Reconstructed tensor
        stats : dict
            Quality statistics
        """
        print("Starting HODMD-TT Analysis")
        print("=" * 50)
        
        # Algorithm 1: TT Decomposition
        tensor = self.tensorize_data(f)
        tt_cores = self.algorithm1_tt_decomposition(tensor)
        
        # Algorithm 2: Orthogonalization
        left_cores, right_cores = self.algorithm2_orthogonalization(tt_cores)
        
        # Algorithm 3: Pseudo Inverse
        pseudo_inv_data = self.algorithm3_pseudo_inverse(left_cores, right_cores)
        
        # Algorithm 4: DMD Computation
        eigenvalues, modes = self.algorithm4_dmd_computation(tensor, pseudo_inv_data)
        
        # Algorithm 5: Reconstruction
        reconstructed = self.algorithm5_reconstruction(tensor, modes, eigenvalues)
        
        # Algorithm 6: Statistics
        stats = self.algorithm6_statistics(tensor, reconstructed)
        
        return reconstructed, stats
    
    def visualize_results(self, figsize=(20, 20)):
        """Comprehensive visualization of HODMD/DMD results including error, eigenvalues, amplitudes, and statistics."""
        if self.reconstructed_data is None:
            self.predict()
        
        fig = plt.figure(figsize=figsize)

        # Original and reconstructed data
        ax1 = plt.subplot(3, 3, 1)
        plt.contourf(np.real(np.reshape(self.original_data[:,0],(449,199))).T, levels = 1001, vmin=-2, vmax=2)
        plt.scatter(50,100,900,color='white', zorder=2) # draw cylinder
        plt.title('Original Data')
        plt.ylabel('Spatial Mode')
        plt.xlabel('Time')

        ax2 = plt.subplot(3, 3, 2)
        plt.contourf(np.real(np.reshape(self.reconstructed_data[:,0],(449,199))).T, levels = 1001, vmin=-2, vmax=2)
        plt.scatter(50,100,900,color='white', zorder=2) # draw cylinder
        plt.title('Reconstructed Data by DMD matrix')
        plt.ylabel('Spatial Mode')
        plt.xlabel('Time')

        ax3 = plt.subplot(3, 3, 3)
        error_data = self.original_data - self.reconstructed_data
        im3 = plt.imshow(error_data, aspect='auto', cmap='RdBu_r')
        plt.title('Reconstruction Error')
        plt.ylabel('Spatial Mode')
        plt.colorbar(im3)

        # Eigenvalue plot
        ax4 = plt.subplot(3, 3, 4)
        plt.scatter(np.real(self.eigenvalues), np.imag(self.eigenvalues),
                    c=np.abs(self.eigenvalues), cmap='plasma', s=60, alpha=0.7)
        plt.colorbar(label='|λ|')
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='Unit Circle')
        plt.xlabel('Real(λ)')
        plt.ylabel('Imag(λ)')
        plt.title('Eigenvalues')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')

        # Mode amplitudes
        ax5 = plt.subplot(3, 3, 5)
        plt.semilogy(np.abs(self.amplitudes), 'bo-', markersize=4)
        plt.xlabel('Mode Index')
        plt.ylabel('|Amplitude|')
        plt.title('Mode Amplitudes')
        plt.grid(True, alpha=0.3)

        # Time series comparison
        ax6 = plt.subplot(3, 3, 6)
        mid_spatial = self.original_data.shape[0] // 2
        plt.plot(self.original_data[mid_spatial, :], 'b-', label='Original', alpha=0.7)
        plt.plot(self.reconstructed_data[mid_spatial, :], 'r--', label='Reconstructed', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title(f'Time Series (Spatial Mode {mid_spatial})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Rank vs Accuracy (MSE)
        ax7 = plt.subplot(3, 3, 7)
        if hasattr(self, 'benchmark_data') and 'rank' in self.benchmark_data:
            ranks = self.benchmark_data['rank']
            mses = self.benchmark_data['rmse']
            plt.plot(ranks, mses, 'go-', lw=2)
            plt.xlabel('SVD Rank')
            plt.ylabel('RMSE')
            plt.title('Rank vs Accuracy (RMSE)')
            plt.grid(True, alpha=0.3)
            plt.yscale("log")

        # Speed vs Accuracy (GPU only)
        ax8 = plt.subplot(3, 3, 8)
        if hasattr(self, 'benchmark_data') and 'time' in self.benchmark_data:
            times = self.benchmark_data['time']
            mses = self.benchmark_data['rmse']
            plt.plot(times, mses, 'mo-', lw=2)
            plt.xlabel('Time (s)')
            plt.ylabel('MSE')
            plt.title('Speed vs Accuracy')
            plt.grid(True, alpha=0.3)
            plt.yscale("log")

        # Rank vs Speed
        ax9 = plt.subplot(3, 3, 9)
        if hasattr(self, 'benchmark_data') and 'time' in self.benchmark_data:
            ranks = self.benchmark_data['rank']
            times = self.benchmark_data['time']
            plt.plot(ranks, times, 'co-', lw=2)
            plt.xlabel('SVD Rank')
            plt.ylabel('Time (s)')
            plt.title('Rank vs Computation Time')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    def benchmark_analysis(self, X, rank_list=None):
        """
        Benchmark DMD/HODMD at different SVD ranks.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data matrix
        rank_list : list
            List of SVD ranks to test
        
        Returns:
        --------
        dict with keys: 'rank', 'mse', 'rmse', 'time'
        """
        if rank_list is None:
            rank_list = [1, 2, 5, 10, 20, 50]
        
        benchmark_data = {
            'rank': [],
            'mse': [],
            'rmse': [],
            'time': []
        }

        print("Starting benchmark analysis...")
        for rank in rank_list:
            print(f"Testing SVD rank: {rank}")
            self.svd_rank = rank
            start_time = time.time()
            self.fit(X)
            self.predict()
            stats = self.compute_statistics()
            elapsed = time.time() - start_time
            
            benchmark_data['rank'].append(rank)
            benchmark_data['mse'].append(stats['mse'])
            benchmark_data['rmse'].append(stats['rmse'])
            benchmark_data['time'].append(elapsed)

        self.benchmark_data = benchmark_data
        print("Benchmark completed.")
        return benchmark_data


class DMD_TT:
    """
    Higher-Order Dynamic Mode Decomposition using Tensor Train decomposition
    """

    def __init__(self, rank=None, svd_rank=None, gpu=False, precision='double'):
        """
        Initialize HODMD with TT decomposition

        Parameters:
        -----------
        rank : list or int
            TT ranks for decomposition
        svd_rank : int
            SVD rank for DMD computation
        gpu : bool
            Use GPU acceleration if available
        precision : str
            'single', 'double', or 'highest'
        """
        self.rank = rank
        self.svd_rank = svd_rank
        self.gpu = gpu and TORCH_AVAILABLE
        self.precision = precision
        self.device = None
        self.dtype_np = None
        self.dtype_torch = None

        self._setup_precision()
        self._setup_device()

        self.tt_cores = None
        self.dmd_modes = None
        self.eigenvalues = None
        self.reconstruction_error = None

    def _setup_precision(self):
        """Setup precision types"""
        if self.precision == 'single':
            self.dtype_np = np.float32
            self.dtype_torch = torch.float32
        elif self.precision == 'double':
            self.dtype_np = np.float64
            self.dtype_torch = torch.float64
        else:  # highest
            self.dtype_np = np.longdouble
            self.dtype_torch = torch.float64  # PyTorch doesn't support long double

    def _setup_device(self):
        """Setup computation device"""
        if self.gpu and TORCH_AVAILABLE:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = torch.device('cpu')
                self.gpu = False
                print("CUDA not available, falling back to CPU")
        else:
            self.device = torch.device('cpu') if TORCH_AVAILABLE else None

    def _to_tensor(self, array):
        """Convert numpy array to torch tensor with proper device and dtype"""
        if self.gpu and TORCH_AVAILABLE:
            return torch.from_numpy(array.astype(self.dtype_np)).to(self.device, dtype=self.dtype_torch)
        return array.astype(self.dtype_np)

    def _to_numpy(self, tensor):
        """Convert tensor back to numpy"""
        if self.gpu and TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy().astype(self.dtype_np)
        return tensor.astype(self.dtype_np)

    def tensorize_data(self, f):
        """
        Convert 3D data to tensor format

        Parameters:
        -----------
        f : np.ndarray
            Input data of shape (nx, ny, nt)

        Returns:
        --------
        tensor : np.ndarray
            Tensorized data
        """
        print(f"Input data shape: {f.shape}")
        return f

    def algorithm1_tt_decomposition(self, tensor):
        """
        Algorithm 1: Convert tensor into TT-format using low-rank decomposition

        Parameters:
        -----------
        tensor : np.ndarray
            Input tensor

        Returns:
        --------
        tt_cores : list
            List of TT cores
        """
        print("Algorithm 1: TT Decomposition")

        # Set default ranks if not provided
        if self.rank is None:
            # Adaptive rank selection based on tensor dimensions
            dims = tensor.shape
            max_rank = min(50, min(dims)//2)
            self.rank = [1] + [max_rank] * (len(dims)-1) + [1]

        # Perform TT decomposition using tensorly
        try:
            tt_tensor = tensor_train(tensor, rank=self.rank)
            self.tt_cores = tt_tensor.factors
            print(f"TT decomposition successful with ranks: {[core.shape for core in self.tt_cores]}")
            return self.tt_cores
        except Exception as e:
            print(f"TensorLy decomposition failed, using manual SVD-based approach: {e}")
            return self._manual_tt_decomposition(tensor)

    def _manual_tt_decomposition(self, tensor):
        """
        Manual TT decomposition using SVD
        """
        cores = []
        current_tensor = tensor.copy()

        for i in range(len(tensor.shape) - 1):
            # Reshape for SVD
            shape = current_tensor.shape
            matrix = current_tensor.reshape(shape[0], -1)

            # SVD
            U, s, Vt = svd(matrix, full_matrices=False)

            # Determine rank
            if isinstance(self.rank, list):
                r = min(self.rank[i+1], len(s))
            else:
                r = min(self.rank or 10, len(s))

            # Truncate
            U = U[:, :r]
            s = s[:r]
            Vt = Vt[:r, :]

            # Store core
            if i == 0:
                cores.append(U.reshape(1, shape[0], r))
            else:
                cores.append(U.reshape(U.shape[0], shape[0]//U.shape[0], r))

            # Update tensor for next iteration
            current_tensor = (np.diag(s) @ Vt).reshape((r,) + shape[1:])

        # Last core
        cores.append(current_tensor.reshape(current_tensor.shape + (1,)))

        self.tt_cores = cores
        print(f"Manual TT decomposition completed with {len(cores)} cores")
        return cores

    def algorithm2_orthogonalization(self, tt_cores):
        """
        Algorithm 2: Left and right orthogonalization of TT cores

        Parameters:
        -----------
        tt_cores : list
            List of TT cores

        Returns:
        --------
        left_ortho_cores : list
            Left-orthogonalized cores
        right_ortho_cores : list
            Right-orthogonalized cores
        """
        print("Algorithm 2: Orthogonalization")

        n_cores = len(tt_cores)
        left_ortho_cores = [None] * n_cores
        right_ortho_cores = [None] * n_cores

        # Left orthogonalization
        for i in range(n_cores - 1):
            core = tt_cores[i]
            r_left, n_i, r_right = core.shape

            # Reshape for QR decomposition
            matrix = core.reshape(r_left * n_i, r_right)
            Q, R = qr(matrix, mode='economic')

            # Update current core
            left_ortho_cores[i] = Q.reshape(r_left, n_i, Q.shape[1])

            # Update next core
            if i < n_cores - 1:
                next_core = tt_cores[i + 1]
                r_left_next, n_next, r_right_next = next_core.shape
                next_matrix = next_core.reshape(r_left_next, n_next * r_right_next)
                updated_next = R @ next_matrix
                tt_cores[i + 1] = updated_next.reshape(R.shape[0], n_next, r_right_next)

        left_ortho_cores[-1] = tt_cores[-1]

        # Right orthogonalization
        tt_cores_copy = [core.copy() for core in left_ortho_cores]

        for i in range(n_cores - 1, 0, -1):
            core = tt_cores_copy[i]
            r_left, n_i, r_right = core.shape

            # Reshape for QR decomposition
            matrix = core.reshape(r_left, n_i * r_right)
            Q, R = qr(matrix.T, mode='economic')
            Q = Q.T
            R = R.T

            # Update current core
            right_ortho_cores[i] = Q.reshape(Q.shape[0], n_i, r_right)

            # Update previous core
            if i > 0:
                prev_core = tt_cores_copy[i - 1]
                r_left_prev, n_prev, r_right_prev = prev_core.shape
                prev_matrix = prev_core.reshape(r_left_prev * n_prev, r_right_prev)
                updated_prev = prev_matrix @ R
                tt_cores_copy[i - 1] = updated_prev.reshape(r_left_prev, n_prev, R.shape[1])

        right_ortho_cores[0] = tt_cores_copy[0]

        print("Orthogonalization completed")
        return left_ortho_cores, right_ortho_cores

    def algorithm3_pseudo_inverse(self, left_cores, right_cores):
        """
        Algorithm 3: Pseudo inverse of matricization of TT-cores tensor

        Parameters:
        -----------
        left_cores : list
            Left-orthogonalized cores
        right_cores : list
            Right-orthogonalized cores

        Returns:
        --------
        pseudo_inv : np.ndarray
            Pseudo inverse matrix
        """
        print("Algorithm 3: Pseudo Inverse Computation")

        n_cores = len(left_cores)
        middle_idx = n_cores // 2

        # SVD on middle core
        middle_core = left_cores[middle_idx]
        r_left, n_i, r_right = middle_core.shape
        matrix = middle_core.reshape(r_left, n_i * r_right)

        U, s, Vt = svd(matrix, full_matrices=False)

        # Truncate based on svd_rank
        if self.svd_rank:
            rank = min(self.svd_rank, len(s))
            U = U[:, :rank]
            s = s[:rank]
            Vt = Vt[:rank, :]

        # Compute pseudo inverse
        s_inv = np.zeros_like(s)
        s_inv[s > 1e-12] = 1.0 / s[s > 1e-12]
        pseudo_inv = Vt.T @ np.diag(s_inv) @ U.T

        print(f"Pseudo inverse computed with effective rank: {np.sum(s > 1e-12)}")
        return pseudo_inv, U, s, Vt

    def algorithm4_dmd_computation(self, tensor, pseudo_inv_data):
        """
        Algorithm 4: Computation of DMD modes and eigenvalues in TT-format

        Parameters:
        -----------
        tensor : np.ndarray
            Original tensor
        pseudo_inv_data : tuple
            Pseudo inverse computation results

        Returns:
        --------
        eigenvalues : np.ndarray
            DMD eigenvalues
        modes : np.ndarray
            DMD modes
        """
        print("Algorithm 4: DMD Computation")

        pseudo_inv, U, s, Vt = pseudo_inv_data

        # Create data matrices for DMD
        nx, ny, nt = tensor.shape

        # Reshape tensor to matrix form for DMD
        X = tensor.reshape(nx * ny, nt)
        X1 = X[:, :-1]  # Data matrix
        X2 = X[:, 1:]   # Shifted data matrix

        # DMD computation
        # Compute A_tilde = U^T * X2 * X1^T * U * S^{-1}
        try:
            # SVD of X1 for better numerical stability
            U_x, s_x, Vt_x = svd(X1, full_matrices=False)

            # Truncate based on energy or rank
            energy_threshold = 0.99
            cumulative_energy = np.cumsum(s_x**2) / np.sum(s_x**2)
            effective_rank = np.argmax(cumulative_energy >= energy_threshold) + 1
            effective_rank = min(effective_rank, len(s_x))

            U_x = U_x[:, :effective_rank]
            s_x = s_x[:effective_rank]
            Vt_x = Vt_x[:effective_rank, :]

            # Compute reduced-order DMD
            A_tilde = U_x.T @ X2 @ Vt_x.T @ np.diag(1.0/s_x)

            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eig(A_tilde)

            # Compute DMD modes
            modes = X2 @ Vt_x.T @ np.diag(1.0/s_x) @ eigenvectors

            print(f"DMD computation completed with {len(eigenvalues)} modes")
            print(f"Eigenvalue magnitudes range: [{np.min(np.abs(eigenvalues)):.6f}, {np.max(np.abs(eigenvalues)):.6f}]")

        except Exception as e:
            print(f"Advanced DMD failed, using simple approach: {e}")
            # Fallback to simple DMD
            A_approx = X2 @ pinv(X1)
            eigenvalues, modes = eigs(A_approx, k=min(10, min(X1.shape)-1))

        self.eigenvalues = eigenvalues
        self.dmd_modes = modes

        return eigenvalues, modes

    def algorithm5_reconstruction(self, original_tensor, modes, eigenvalues, dt=1.0):
        """
        Algorithm 5: Reconstruction of data and visualization

        Parameters:
        -----------
        original_tensor : np.ndarray
            Original tensor data
        modes : np.ndarray
            DMD modes
        eigenvalues : np.ndarray
            DMD eigenvalues
        dt : float
            Time step

        Returns:
        --------
        reconstructed_tensor : np.ndarray
            Reconstructed tensor
        """
        print("Algorithm 5: Data Reconstruction")

        nx, ny, nt = original_tensor.shape

        # Time vector
        t = np.arange(nt) * dt

        # Compute time dynamics
        time_dynamics = np.zeros((len(eigenvalues), nt), dtype=complex)
        for i, lam in enumerate(eigenvalues):
            time_dynamics[i, :] = np.power(lam, t)

        # Initial conditions (project first snapshot onto modes)
        X_flat = original_tensor.reshape(nx * ny, nt)
        b = pinv(modes) @ X_flat[:, 0]

        # Reconstruct
        X_dmd = modes @ np.diag(b) @ time_dynamics

        # Take real part and reshape
        X_reconstructed = np.real(X_dmd).reshape(nx, ny, nt)

        # Compute reconstruction error
        self.reconstruction_error = np.linalg.norm(original_tensor - X_reconstructed) / np.linalg.norm(original_tensor)

        print(f"Reconstruction completed with relative error: {self.reconstruction_error:.6f}")

        return X_reconstructed

    def algorithm6_statistics(self, original_tensor, reconstructed_tensor):
        """
        Algorithm 6: Statistics to assess reconstruction quality

        Parameters:
        -----------
        original_tensor : np.ndarray
            Original tensor
        reconstructed_tensor : np.ndarray
            Reconstructed tensor

        Returns:
        --------
        stats : dict
            Dictionary containing various quality metrics
        """
        print("Algorithm 6: Quality Assessment")

        # Various error metrics
        print(f"Shape of original tensor: {original_tensor.shape}")
        print(f"Shape of reconstructed tensor: {reconstructed_tensor.shape}")
        frobenius_error = np.linalg.norm(original_tensor - reconstructed_tensor)
        relative_error = frobenius_error / np.linalg.norm(original_tensor)

        # Energy preservation
        original_energy = np.sum(original_tensor**2)
        reconstructed_energy = np.sum(reconstructed_tensor**2)
        energy_ratio = reconstructed_energy / original_energy

        # Correlation coefficient
        orig_flat = original_tensor.flatten()
        recon_flat = reconstructed_tensor.flatten()
        correlation = np.corrcoef(orig_flat, recon_flat)[0, 1]

        # Eigenvalue analysis
        stable_modes = np.sum(np.abs(self.eigenvalues) <= 1.0)
        unstable_modes = np.sum(np.abs(self.eigenvalues) > 1.0)

        stats = {
            'frobenius_error': frobenius_error,
            'relative_error': relative_error,
            'energy_ratio': energy_ratio,
            'correlation': correlation,
            'stable_modes': stable_modes,
            'unstable_modes': unstable_modes,
            'total_modes': len(self.eigenvalues),
            'effective_rank': np.sum(np.abs(self.eigenvalues) > 1e-10)
        }

        print(f"Quality Statistics:")
        print(f"  Relative Error: {relative_error:.6f}")
        print(f"  Energy Ratio: {energy_ratio:.6f}")
        print(f"  Correlation: {correlation:.6f}")
        print(f"  Stable/Unstable Modes: {stable_modes}/{unstable_modes}")

        return stats

    def fit(self, f):
        """
        Complete HODMD-TT pipeline

        Parameters:
        -----------
        f : np.ndarray
            Input tensor of shape (nx, ny, nt)

        Returns:
        --------
        reconstructed : np.ndarray
            Reconstructed tensor
        stats : dict
            Quality statistics
        """
        print("Starting HODMD-TT Analysis")
        print("=" * 50)

        # Algorithm 1: TT Decomposition
        tensor = self.tensorize_data(f)
        tt_cores = self.algorithm1_tt_decomposition(tensor)

        # Algorithm 2: Orthogonalization
        left_cores, right_cores = self.algorithm2_orthogonalization(tt_cores)

        # Algorithm 3: Pseudo Inverse
        pseudo_inv_data = self.algorithm3_pseudo_inverse(left_cores, right_cores)

        # Algorithm 4: DMD Computation
        eigenvalues, modes = self.algorithm4_dmd_computation(tensor, pseudo_inv_data)

        # Algorithm 5: Reconstruction
        reconstructed = self.algorithm5_reconstruction(tensor, modes, eigenvalues)

        # Algorithm 6: Statistics
        stats = self.algorithm6_statistics(tensor, reconstructed)

        return reconstructed, stats
    def predict(self, time_steps=None):
        """
        Predict/reconstruct data using DMD with Tensor Train decomposition

        Parameters:
        -----------
        time_steps : int or None
            Number of time steps to predict

        Returns:
        --------
        reconstructed_data : np.ndarray
            Reconstructed tensor data
        """
        if self.dmd_modes is None or self.eigenvalues is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Determine the number of time steps
        if time_steps is None:
            # Use the original tensor's time dimension if available
            if hasattr(self, 'original_tensor'):
                time_steps = self.original_tensor.shape[2]
            else:
                raise ValueError("Original tensor not available. Provide time_steps.")

        # Time dynamics
        time_dynamics = np.zeros((len(self.eigenvalues), time_steps), dtype=complex)
        for i, lam in enumerate(self.eigenvalues):
            time_dynamics[i, :] = np.power(lam, np.arange(time_steps))

        # Initial conditions (project first snapshot onto modes)
        nx, ny = self.original_tensor.shape[0], self.original_tensor.shape[1]
        X_flat = self.original_tensor.reshape(nx * ny, -1)
        b = np.linalg.pinv(self.dmd_modes) @ X_flat[:, 0]

        # Reconstruct
        X_dmd = self.dmd_modes @ np.diag(b) @ time_dynamics

        # Take real part and reshape
        self.reconstructed_data = np.real(X_dmd).reshape(nx, ny, time_steps)

        return self.reconstructed_data


    def visualize_results(self, figsize=(20, 20)):
        """Comprehensive visualization of HODMD/DMD results including error, eigenvalues, amplitudes, and statistics."""
        if self.reconstructed_data is None:
            self.predict()

        fig = plt.figure(figsize=figsize)

        # Original and reconstructed data
        ax1 = plt.subplot(3, 3, 1)
        plt.contourf(np.real(np.reshape(self.original_data[:,0],(449,199))).T, levels = 1001, vmin=-2, vmax=2)
        plt.scatter(50,100,900,color='white', zorder=2) # draw cylinder
        plt.title('Original Data')
        plt.ylabel('Spatial Mode')
        plt.xlabel('Time')

        ax2 = plt.subplot(3, 3, 2)
        plt.contourf(np.real(np.reshape(self.reconstructed_data[:,0],(449,199))).T, levels = 1001, vmin=-2, vmax=2)
        plt.scatter(50,100,900,color='white', zorder=2) # draw cylinder
        plt.title('Reconstructed Data by DMD matrix')
        plt.ylabel('Spatial Mode')
        plt.xlabel('Time')

        ax3 = plt.subplot(3, 3, 3)
        error_data = self.original_data - self.reconstructed_data
        im3 = plt.imshow(error_data, aspect='auto', cmap='RdBu_r')
        plt.title('Reconstruction Error')
        plt.ylabel('Spatial Mode')
        plt.colorbar(im3)

        # Eigenvalue plot
        ax4 = plt.subplot(3, 3, 4)
        plt.scatter(np.real(self.eigenvalues), np.imag(self.eigenvalues),
                    c=np.abs(self.eigenvalues), cmap='plasma', s=60, alpha=0.7)
        plt.colorbar(label='|λ|')
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='Unit Circle')
        plt.xlabel('Real(λ)')
        plt.ylabel('Imag(λ)')
        plt.title('Eigenvalues')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')

        # Mode amplitudes
        ax5 = plt.subplot(3, 3, 5)
        plt.semilogy(np.abs(self.amplitudes), 'bo-', markersize=4)
        plt.xlabel('Mode Index')
        plt.ylabel('|Amplitude|')
        plt.title('Mode Amplitudes')
        plt.grid(True, alpha=0.3)

        # Time series comparison
        ax6 = plt.subplot(3, 3, 6)
        mid_spatial = self.original_data.shape[0] // 2
        plt.plot(self.original_data[mid_spatial, :], 'b-', label='Original', alpha=0.7)
        plt.plot(self.reconstructed_data[mid_spatial, :], 'r--', label='Reconstructed', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title(f'Time Series (Spatial Mode {mid_spatial})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Rank vs Accuracy (MSE)
        ax7 = plt.subplot(3, 3, 7)
        if hasattr(self, 'benchmark_data') and 'rank' in self.benchmark_data:
            ranks = self.benchmark_data['rank']
            mses = self.benchmark_data['rmse']
            plt.plot(ranks, mses, 'go-', lw=2)
            plt.xlabel('SVD Rank')
            plt.ylabel('RMSE')
            plt.title('Rank vs Accuracy (RMSE)')
            plt.grid(True, alpha=0.3)
            plt.yscale("log")

        # Speed vs Accuracy (GPU only)
        ax8 = plt.subplot(3, 3, 8)
        if hasattr(self, 'benchmark_data') and 'time' in self.benchmark_data:
            times = self.benchmark_data['time']
            mses = self.benchmark_data['rmse']
            plt.plot(times, mses, 'mo-', lw=2)
            plt.xlabel('Time (s)')
            plt.ylabel('MSE')
            plt.title('Speed vs Accuracy')
            plt.grid(True, alpha=0.3)
            plt.yscale("log")

        # Rank vs Speed
        ax9 = plt.subplot(3, 3, 9)
        if hasattr(self, 'benchmark_data') and 'time' in self.benchmark_data:
            ranks = self.benchmark_data['rank']
            times = self.benchmark_data['time']
            plt.plot(ranks, times, 'co-', lw=2)
            plt.xlabel('SVD Rank')
            plt.ylabel('Time (s)')
            plt.title('Rank vs Computation Time')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def benchmark_analysis(self, X, rank_list=None):
        """
        Benchmark DMD/HODMD at different SVD ranks.

        Parameters:
        -----------
        X : np.ndarray
            Input data matrix
        rank_list : list
            List of SVD ranks to test

        Returns:
        --------
        dict with keys: 'rank', 'mse', 'rmse', 'time'
        """
        if rank_list is None:
            rank_list = [1, 2, 5, 10, 20, 50]

        benchmark_data = {
            'rank': [],
            'mse': [],
            'rmse': [],
            'time': []
        }

        print("Starting benchmark analysis...")
        for rank in rank_list:
            print(f"Testing SVD rank: {rank}")
            self.svd_rank = rank
            start_time = time.time()
            self.fit(X)
            self.predict()
            stats = self.compute_statistics()
            elapsed = time.time() - start_time

            benchmark_data['rank'].append(rank)
            benchmark_data['mse'].append(stats['mse'])
            benchmark_data['rmse'].append(stats['rmse'])
            benchmark_data['time'].append(elapsed)

        self.benchmark_data = benchmark_data
        print("Benchmark completed.")
        return benchmark_data
