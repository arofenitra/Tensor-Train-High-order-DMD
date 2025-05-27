# Tensor-Train-High-order-DMD

This project focuses on the Tensor Train Implementation of DMD and HODMD algorithms.

## Repository Structure

.
├── src
│   └── lib.py                # Contains all functions and imports
├── .Trash_and_failed_implementation
│   └── ...                  # Contains failed results
├── .vscode
├── notebook
│   └── ...                  # Contains all used .ipynb files
├── .jupyter_ystore.db
├── README.md
├── requirements.txt
├── dataset
│   └── VORTAL.mat           # Dataset used in the project
└── docs
└── ...                 # Contains all used PDF implementations





## Tensor Train

### Introduction and Motivations for Tensor Train

#### What is a Tensor?

- **0-dimensional tensor**: A point (tensor of order 0)
- **1-dimensional tensor**: A vector (tensor of order 1)
- **2-dimensional tensor**: A matrix (tensor of order 2)
- **3-dimensional tensor**: Example: A color RGB image (tensor of order 3)
- **4-dimensional tensor**: Example: A batch of color RGB images or a video (tensor of order 4)

#### Tensor Train Overview

A tensor \( A \in \mathbb{R}^{n_1 \times \cdots \times n_d} \) is represented in the **TT format** as:
\[ A(i_1, \cdots, i_d) = G_1(i_1) \cdots G_d(i_d) \]

Each \( G_k(i_k) \in \mathbb{R}^{r_{k-1} \times r_k} \) is called a **TT-core**, where \( r_0 = 1, r_d = 1 \), and \( r_k \) are the TT-ranks.

#### Problems with High Dimensional Tensors

- **Curse of Dimensionality and Storage**: The number of elements grows exponentially with the order of the tensor.
- **Computation**: The number of operations grows exponentially with the order of the tensor.
- **Visualization**: It is impossible to visualize tensors with an order greater than 3.

#### What is a Tensor Train (TT)?

- It decomposes a high-dimensional tensor into a sequence of 3D tensors, called cores (TT-cores). This reduces computational cost and storage drastically.
- For a tensor \( \mathcal{T} \in \mathbb{R}^{n_1 \times n_2 \times \ldots \times n_d} \), its TT-decomposition is:

\[ \mathcal{T}(i_1, \ldots, i_d) = G^{(1)}(i_1) \cdot G^{(2)}(i_2) \cdot \ldots \cdot G^{(d)}(i_d) \]

where each \( G^{(k)} \) is a 3D tensor of size \( r_{k-1} \times n_k \times r_k \) and \( r_0 = r_d = 1 \). The numbers \( r_k \) are called TT-ranks.

### TT-Ranks, Compression

#### TT-Ranks

- Controls the storage cost, computational complexity, and accuracy of the tensor. Low TT-ranks have efficient representation but low accuracy.
- TT-ranks are the dimensions of the core tensors.

#### TT Compression

- **TT-SVD**: Sequential SVD-based compression
- **TT-Cross**: Interpolation-approximation based compression

#### TT-SVD Algorithm Decomposition

For a tensor \( \mathcal{T} \in \mathbb{R}^{n_1 \times n_2 \times \ldots \times n_d} \), to find the core tensors \( G^{(k)} \) and the TT-ranks \( r_k \):

\[ \mathcal{T}(i_1, \ldots, i_d) = G^{(1)}(i_1) \cdot G^{(2)}(i_2) \cdot \ldots \cdot G^{(d)}(i_d) \]

The TT-SVD algorithm is as follows:

1. **Step 1**: Unfolding into mod-1 tensor
   \[ \mathcal{T} \rightarrow \mathcal{T}_{(1)} \in \mathbb{R}^{n_1 \times (n_2 \times \ldots \times n_d)} \]

2. **Step 2**: SVD on mod-1 tensor
   \[ \mathcal{T}_{(1)} = U_1 \Sigma_1 V_1^T \]
   where \( U_1 \in \mathbb{R}^{n_1 \times r_1} \), \( \Sigma_1 \in \mathbb{R}^{r_1 \times r_1} \), and \( V_1 \in \mathbb{R}^{(n_2 \times \ldots \times n_d) \times r_1} \)
   \[ G^{(1)} = \text{reshape } U_1 \text{ into } r_0 \times n_1 \times r_1 \text{ and } \mathcal{T}_{(1)}' = \Sigma_1 \cdot V_1^T \]

3. **Step 3**: Reshaping into mod-1,2 tensor
   \[ \mathcal{T}_{(1)}' \rightarrow \mathcal{T}_{(2)}' \in \mathbb{R}^{(r_1 n_2) \times (n_3 \times \ldots \times n_d)} \]

4. **Step 4**: SVD on mod-1,2 tensor
   \[ \mathcal{T}_{(2)}' = U_2 \Sigma_2 V_2^T \]
   where \( U_2 \in \mathbb{R}^{(r_1 n_2) \times r_2} \), \( \Sigma_2 \in \mathbb{R}^{r_2 \times r_2} \), and \( V_2 \in \mathbb{R}^{(n_3 \times \ldots \times n_d) \times r_2} \)
   \[ G^{(2)} = \text{reshape } U_2 \text{ into } r_1 \times n_2 \times r_2 \text{ and } \mathcal{T}_{(2)}'' = \Sigma_2 \cdot V_2^T \]

5. **Step 5**: Repeat steps 3 and 4 until the last mode.

### Orthogonalization in TT-Format

#### Left Orthogonalization

We use **QR** decomposition of the tensor to obtain the left orthogonal components. All cores left of position \( k \) are orthogonal:
\[ \sum_{j=1}^{n_j} G_j(i)^T G_j(i) = I \]

**Algorithm:**

- **Inputs**: Cores of the TT-decomposition: \( G_1 \) (dim: \( 1 \times n_1 \times r_1 \)), \( G_2 \) (dim: \( r_1 \times n_2 \times r_2 \)), ...

**Steps:**

a. Take a core \( G_0 \) and reshape it into \( (1 \cdot n_1) \times r_1 \).

b. Perform **SVD** decomposition (or QR) of the current core (here \( G_0 \)), i.e., \( G_0 = (U_0 S_0 V_0^T) \) where \( Q = U_0 : (n_1 \times s_1) \) and \( R = S_0 V_0^T : (s_1 \times r_1) \). Take the 1st core as \( Q : \text{reshape}(1, n_1, s_1) \).

c. Multiply \( R \) to the next cores \( G_1 \), the result will be of shape \( (s_1, n_2, r_2) \). Reshape it into \( (s_1 \cdot n_2, r_2) \).

d. Perform **SVD** decomposition (seen as QR) of the current core (here \( G_1 \)), i.e., \( G_1 = (U_1 S_1 V_1^T) \) where \( Q = U_1 : ((s_1 \cdot n_2) \times s_2) \) and \( R = S_1 V_1^T : (s_2 \times r_2) \). Take the 2nd core as \( Q : \text{reshape}(s_1, n_2, s_2) \).

e. Multiply \( R \) to the next cores \( G_2 \), the result will be of shape \( (s_2, n_3, r_3) \). Reshape it into \( (s_2 \cdot n_3, r_3) \).

f. Repeat the steps until the last core.

The output will be a list of orthogonal cores.

#### Right Orthogonalization

We use **QR** decomposition of the transpose of the tensor to obtain the transpose of orthogonal components. All cores left of position \( k \) are orthogonal:
\[ \sum_{j=1}^{n_j} G_j(i) G_j(i)^T = I \]

## DMD with Tensor Algorithms

### DMD with Tensor Algorithms and its Similarity with DMD Algorithms on Matrix

- Given a sequence of d-dimensional tensors \( X_i \in \mathbb{R}^{n_1 \times n_2 \times \cdots \times n_d} \), stack them into two large tensor datasets \( \mathcal{X} \) and \( \mathcal{Y} \).
- Compute a low-rank TT-format decomposition of these tensor datasets.
- Use properties of the TT-decomposition to compute the pseudoinverse \( X^+ \) of the tensor unfolding \( \mathcal{X} = \text{matricize}(\mathcal{X}) \) without explicitly forming the full matrix.
- Compute the DMD operator \( \mathcal{A} = \mathcal{Y} \mathcal{X}^+ \), and extract its eigenvalues and eigenvectors — the DMD modes and frequencies.

## DMD

### Overview

Let \( X \in \mathbf{R}^{M \times N} \) be a data matrix.

- Each column of \( X \) is a snapshot of the system.
- Each row of \( X \) is a time series of a single variable.

We consider several snapshots, denoted as:
\[ X_{i_1}^{i_2} = [\mathbf{x}_{i_1}, \mathbf{x}_{i_1+1}, \ldots, \mathbf{x}_{i_2}] \]
The full snapshot matrix is \( X = X_{1}^{N} = [\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_{N}] \), where each \( \mathbf{x}_i = \mathbf{x}(t_i) \in \mathbf{R}^{M} \) is a snapshot of the system at time \( t_i = t_1 + (i-1)\Delta t \).

### Assumption of DMD

The standard **DMD** relies on the assumption:
\[ \mathbf{x}_{i+1} = A \mathbf{x}_i \]
\[ \Leftrightarrow X_{2}^{N} = A X_{1}^{N-1} \]

where \( A \) is a linear operator.

If \( A \) is known and has eigendecomposition \( A = \Phi \Lambda \Phi^{-1} \), then the solution to the DMD problem is:
\[ \mathbf{x}(t_i) = A^{(i-1)} \mathbf{x}_1 \]
\[ \Leftrightarrow \mathbf{x}(t_i) = \Phi \Lambda^{(i-1)} \Phi^{-1} \mathbf{x}_1 \]
\[ \Rightarrow \mathbf{x}(t_i) = \sum_{m=1}^M \phi_m e^{(\delta_m + i \omega_m)(i-1)\Delta t} b_m \]

where \( t_1 \) is the initial time, \( \delta_m + i \omega_m = \frac{1}{\Delta t} \log(\lambda_m) \), \( \phi_m = \Phi \mathbf{e}_m \), and \( b_m = \mathbf{e}_m^* \Phi^{-1} \mathbf{x}_1 \).

### Problems with the Above Methods

1. \( A \) is not known.
2. \( A \) is not necessarily diagonalizable.
3. \( A \) is not necessarily invertible.
4. \( A \) is a very large matrix, and hence computationally expensive.

So we need to find a way to compute the DMD modes without knowing \( A \).

### Efficient DMD Algorithm: DMD Modes Without Knowing \( A \)

The DMD algorithm is a method to compute the dominant modes of \( X \). The algorithm is as follows:

1. Compute the SVD of \( X_1^{N-1} \):
   \[ X_1^{N-1} = U \Sigma V^* \]

   We can take low-rank SVD:
   \[ X_1^{N-1} \approx U_r \Sigma_r V_r^* \]

   where \( U_r \) is the first \( r \) columns of \( U \), \( \Sigma_r \) is the first \( r \) rows and columns of \( \Sigma \), and \( V_r \) is the first \( r \) columns of \( V \).

2. Compute the reduced-order model:
   \[ \hat{A} = U_r^* A U_r = U_r^* X_2^{N} V_r \Sigma_r^{-1} \in \mathbb{C}^{r \times r} \]

   where \( X_2^{N} \) is the matrix of snapshots from \( x_2 \) to \( x_N \).

3. Compute the eigenvalues and eigenvectors of \( \hat{A} \):
   \[ \hat{A} W = W \Lambda \]
   \[ \Lambda, W = \{ \lambda_1, \ldots, \lambda_r \}, \{ w_1, \ldots, w_r \} = \text{eig}(\hat{A}) \]

4. Compute the DMD modes:
   \[ \Phi = U_r W \in \mathbb{C}^{M \times r} \]

   where \( \Omega = [\omega_1, \ldots, \omega_r] \).

   We have here:
   \[ A \Phi = U_r (\hat{A} W) = U_r (W \Lambda) = \Phi \Lambda \]

   which shows that \( \Phi \) is the matrix of DMD modes and \( \lambda_1, \ldots, \lambda_r \) are the DMD eigenvalues.

5. Compute the DMD time series:
   \[ \mathbb{x}(t_i) = \Phi \cdot (\Lambda^{i-1}) \cdot \Phi^{-1} \cdot \mathbb{x}(t_1) \in \mathbb{C}^{M \times 1} \]

## HODMD

### Assumption of HODMD

The **Higher Order Dynamic Mode Decomposition (HODMD)** extends standard DMD by incorporating a **delay embedding**, which allows for better detection of multiple frequencies and more robustness in noisy or nonlinear systems.

Instead of assuming a direct relation \( \mathbf{x}_{i+1} = A \mathbf{x}_i \), we assume that the dynamics are governed by a linear operator acting on a **delay-embedded state vector**:

\[ \mathbf{z}_{i+1} = A_d \mathbf{z}_i \]

where \( \mathbf{z}_i = [\mathbf{x}_i, \mathbf{x}_{i+1}, \ldots, \mathbf{x}_{i+d-1}]^\top \) is the delay-embedded state vector of dimension \( Md \), and \( d \) is the delay parameter.

This implies:

\[ \mathcal{X}_{d+1}^{N} = A_d \mathcal{X}_{1}^{N-d} \]

where \( \mathcal{X}_{1}^{N-d} \) and \( \mathcal{X}_{d+1}^{N} \) are delay-embedded matrices constructed from \( X \).

### Problems with Standard DMD

Standard DMD may fail when:

1. The system has **multiple dominant frequencies**.
2. The system is affected by **noise**.
3. The dynamics are **nonlinear**.

HODMD addresses these issues by using delay embedding, allowing us to:

- Extract **multiple frequencies**.
- Improve **accuracy and robustness**.
- Better model complex systems.

### Efficient HODMD Algorithm

#### Step-by-Step Procedure:

1. **Build Delay-Embedded Matrices**

   Given the original data matrix \( X \in \mathbb{R}^{M \times N} \), construct the delay-embedded matrices:

   \[
   \mathcal{X}_1 =
   \begin{bmatrix}
   \mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x}_{N-d} \\
   \mathbf{x}_2 & \mathbf{x}_3 & \cdots & \mathbf{x}_{N-d+1} \\
   \vdots & \vdots & \ddots & \vdots \\
   \mathbf{x}_d & \mathbf{x}_{d+1} & \cdots & \mathbf{x}_N
   \end{bmatrix}
   \in \mathbb{R}^{Md \times N - d + 1}
   \]

   Split into two parts:

   \[ \mathcal{X}_1 = \text{columns } 1 \text{ to } N - d, \quad \mathcal{X}_2 = \text{columns } 2 \text{ to } N - d + 1 \]

2. **Perform SVD on \( \mathcal{X}_1 \)**

   Compute the singular value decomposition:

   \[ \mathcal{X}_1 = U \Sigma V^* \]

   Optionally truncate to rank \( r \):

   \[ U_r = U[:, :r], \quad \Sigma_r = \text{diag}(\Sigma[:r]), \quad V_r = V[:, :r] \]

3. **Compute Reduced Operator**

   Project the dynamics onto the low-rank subspace:

   \[ \hat{A} = U_r^* \mathcal{X}_2 V_r \Sigma_r^{-1} \in \mathbb{C}^{r \times r} \]

4. **Eigen-decomposition of \( \hat{A} \)**

   Solve the eigenvalue problem:

   \[ \hat{A} W = W \Lambda \]

   Where:

   - \( \Lambda = \text{diag}(\lambda_1, \ldots, \lambda_r) \): eigenvalues (complex)
   - \( W = [\mathbf{w}_1, \ldots, \mathbf{w}_r] \): eigenvectors

5. **Compute Full-Space DMD Modes**

   Reconstruct the DMD modes in the full space:

   \[ \Phi = U_r W \in \mathbb{C}^{Md \times r} \]

   To map back to the original space:

   \[ \phi_m^{(original)} = \frac{1}{d} \sum_{k=1}^d \phi_m^{(k)} \]

   Where \( \phi_m^{(k)} \) is the \( k \)-th block of \( \Phi \mathbf{e}_m \), corresponding to the \( k \)-th delay component.

6. **Reconstruct Time Series**

   Using the initial amplitude coefficients \( b \), computed via least squares:

   \[ b = \arg\min_{\mathbf{b}} \left\| \Phi \mathbf{b} - \mathcal{X}_1[:, 0] \right\| \]

   Then the solution is:

   \[ \mathbf{x}(t_i) = \sum_{m=1}^r b_m \phi_m e^{(\delta_m + i \omega_m)(i-1)\Delta t} \]

   Where:

   - \( \log(\lambda_m) = (\delta_m + i \omega_m) \Delta t \)
   - \( \phi_m \): spatial mode
   - \( b_m \): amplitude coefficient
