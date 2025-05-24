# Tensor-Train-High-order-DMD
This is a project for Tensor Train Implementation of DMD and HODMD algorithms
# Tensor Train

## Introduction and motivations for tensor train
### what is a tensor ?
- A point is a 0-dimensional tensor (aka tensor of order 0)
- A vector is a 1-dimensional tensor (aka tensor of order 1)
- A matrix is a 2-dimensional tensor (aka tensor of order 2)
- A 3-dimensional tensor is a tensor of order 3. Example: a color RGB image is a 3-dimensional tensor
- A 4-dimensional tensor is a tensor of order 4. Example: a batch of color RGB images is a 4-dimensional tensor, or a video.
- etc.
### Tensor Train overview
A tensor $A∈\mathbb{R}^{n_1 \times \cdots n_d}$ is represented in the **TT format** as:
$$A(i_1,\cdots,i_d) = G_1(i_1)\cdots G_d(i_d) $$
Each $G_k(i_k)\in \mathbb{R}^{r_{k-1}}\times r_k$ is called a **TT-core** , where $r_0=1,r_d=1$, and $r_k$ are the TT-ranks .
### Problems with high dimensional tensors
- **Curse of dimensionality and Storage**: the number of elements grows exponentially with the order of the tensor
- **Computation**: the number of operations grows exponentially with the order of the tensor
- **Visualization**: it is impossible to visualize tensors with order greater than 3
- ## What is a Tensor Train (TT)
- It decomposes a high dimensional tensor into a sequence of 3D tensors, which are called cores (TT-cores). It reduces the computational cost and storage drastically.
- For a tensor $\mathcal{T} \in \mathbb{R}^{n_1 \times n_2 \times \ldots \times n_d}$, its TT-decompositions is: 

$$\mathcal{T}(i_1,...,i_d) = G^{(1)}(i_1) \cdot G^{(2)}(i_2) \cdot \ldots \cdot G^{(d)}(i_d)$$
where each $G^{(k)}$ is a 3D tensor of size $r_{k-1} \times n_k \times r_k$ and $r_0 = r_d = 1$. The numbers $r_k$ are called TT-ranks.

## TT-Ranks, Compression
### TT-Ranks
- It controls the storage cost, computational complexity and accuracy of the tensor. Low TT-ranks has efficient representation but low accuracy.
- TT-ranks are the dimensions of the core tensors.
### TT compression
- TT-SVD (sequential SVD-based compression)
- TT-Cross (interpolation-approximation based compression)
### TT_SVD algorithms decomposition
For a tensor $\mathcal{T} \in \mathbb{R}^{n_1 \times n_2 \times \ldots \times n_d}$. To find the core tensors $G^{(k)}$ and the TT-ranks $r_k$,
$$\mathcal{T}(i_1,...,i_d) = G^{(1)}(i_1) \cdot G^{(2)}(i_2) \cdot \ldots \cdot G^{(d)}(i_d)$$
its TT-SVD algorithm is as follow is: 
- **Step 1**: Unfolding into mod-1 tensor 
$\mathcal{T} \rightarrow \mathcal{T}_{(1)} \in \mathbb{R}^{n_1 \times (n_2 \times \ldots \times n_d)}$

- **Step 2**: SVD on mod-1 tensor  
1. $\mathcal{T}_{(1)} = U_1 \Sigma_1 V_1^T$, where $U_1 \in \mathbb{R}^{n_1 \times r_1}$, $\Sigma_1 \in \mathbb{R}^{r_1 \times r_1}$, and $V_1 \in \mathbb{R}^{(n_2 \times \ldots \times n_d) \times r_1}$
2. $G^{(1)} =  \text{ reshape } U_1  \text{ into } r_0 \times n_1 \times r_1 \text{ and } \mathcal{T}_{(1)}' = \Sigma_1 \cdot V_1^T$

- **Step 3**: Reshaping into mod-1,2 tensor

$$\mathcal{T}_{(1)}' \rightarrow \mathcal{T}_{(2)}' \in \mathbb{R}^{(r_1 n_2) \times (n_3 \times \ldots \times n_d)}$$ 
- **Step 4**: SVD on mod-1,2 tensor  
1. $\mathcal{T}_{(2)}' = U_2 \Sigma_2 V_2^T$, where $U_2 \in \mathbb{R}^{(r_1 n_2) \times r_2}$, $\Sigma_2 \in \mathbb{R}^{r_2 \times r_2}$, and $V_2 \in \mathbb{R}^{(n_3 \times \ldots \times n_d) \times r_2}$
2. $G^{(2)} = $ reshape $U_2 $ into $r_1 \times n_2 \times r_2$ and $\mathcal{T}_{(2)}'' = \Sigma_2 \cdot V_2^T$
- **Step 5**: Repeat step 3 and 4 until the last mode
### Orthogonalization in TT-format
1. Left orthogonalization

We uses **QR** decomposition of the tensor to obtain the left orthogonal components.
All cores left of position k are orthogonal:
$$ \sum_{j=1}^{n_j} G_j(i)^TG_j(i) = I$$
Algorithms:  

Inputs : Cores of the TT-decomposition : $G_1$ (dim: $1\times n_1 \times r_1$), $G_2$(dim: $r_1 \times n_2 \times r_2 $),...  
**a.** Take a core $G_0$ and reshape it into $(1*n_1)\times r_1$  
**b.** Perform **SVD** decomposition( or QR) of current core (here $G_0$), i.e. $G_0=(U_0S_0V_0^T)$ where $Q=U_0:(n_1\times s_1)$ and $R=S_0V_0^T: (s_1 \times r_1)$ and takes the 1st core as $Q:$ reshape$(1,n_1,s_1)$  
**c.** Multiply $R$ to the next cores $G_1$, the result will be of shape $(s_1,n_2,r_2)$.  Reshape it into $(s_1*n_2,r_2)$
**d.** Perform **SVD** decomposition(seen as QR) of current core (here $G_1$), i.e. $G_1=(U_1S_1V_1^T)$ where $Q=U_1:((s_1*n_2)\times s_2)$ and $R=S_1V_1^T: (s_2 \times r_2)$ and takes the 2nd core as $Q:$ reshape$(s_1,n_2,s_2)$
**e.** Multiply $R$ to the next cores $G_2$, the result will be of shape $(s_2,n_3,r_3)$.  Reshape it into $(s_2*n_3,r_3)$
**f.** Repeat the steps until the last core.
The output will be a list of orthogonal cores.


2. Right orthogonalization

We uses **QR** decomposition of the transpose of the tensor to obtain the transpose of orthogonal components.
All cores left of position k are orthogonal:
$$ \sum_{j=1}^{n_j} G_j(i)G_j(i)^T = I$$
# DMD with tensor algorithms :
## DMD with tensor algorithms and its Similarity with **DMD** algorithms on matrix:
- Given a sequence of d-dimensional tensors  $X_i​∈\mathbb{R}^{n_1​×n_2​×⋯×_{n_d}}$​, stack them into two large tensor datasets $\mathcal{X}$ and $\mathcal{Y}$.
- Compute a low-rank TT-format decomposition  of these tensor datasets.
- Use properties of the TT-decomposition to compute the pseudoinverse  of the tensor unfolding $\mathcal{X}=\text{matricize}(\mathcal{X})$ without explicitly forming the full matrix.
- Compute the DMD operator $\mathcal{A}=\mathcal{Y}\mathcal{X}^+$, and extract its eigenvalues and eigenvectors — the DMD modes and frequencies.
     
