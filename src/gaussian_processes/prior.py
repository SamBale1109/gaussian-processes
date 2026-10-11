r"""Prior distributions for Gaussian processes."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from gaussian_processes.kernels import rbf_kernel

def sample_from_multivariate_gaussian(mu: np.ndarray, K: np.ndarray, n_samples: int = 10, jitter=1e-10, rng: np.random.Generator | None = None):
    r"""Draw samples from a multivariate Gaussian distribution.

    A two-dimensional Gaussian random vector is defined by its
    mean vector and covariance matrix:

    \f[
        \mathbf{X} \sim \mathcal{N}
        \left(
            \begin{bmatrix}
                \mu_1 \\
                \mu_2
            \end{bmatrix},
            \begin{bmatrix}
                \Sigma_{11} & \Sigma_{12} \\
                \Sigma_{21} & \Sigma_{22}
            \end{bmatrix}
        \right)
    \f]

    To generate correlated Gaussian samples, the covariance
    matrix is decomposed using the Cholesky decomposition:

    \f[
        \Sigma = LL^\top
    \f]

    A sample can then be constructed as:

    \f[
        \mathbf{X} = \boldsymbol{\mu}
        + L\mathbf{Z},
        \qquad
        \mathbf{Z} \sim \mathcal{N}(\mathbf{0}, I)
    \f]

    For numerical stability, a small jitter value is added
    to the diagonal of the covariance matrix before decomposition.

    \param mu Mean vector of shape (n_points,).
    \param K Covariance matrix of shape (n_points, n_points).
    \param n_samples Number of samples to generate.
    \param jitter Non-negative diagonal regularization for numerical stability.
    \param rng NumPy random generator. If None, a new generator is created.

    \return Array of shape (n_points, n_samples), with one sample per column.

    \throws ValueError If the input parameters are invalid.
    \throws numpy.linalg.LinAlgError If Cholesky decomposition fails.
    """

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("Covariance matrix must be square.")

    if mu.ndim != 1 or mu.shape[0] != K.shape[0]:
        raise ValueError("Mean vector has incompatible shape.")

    if n_samples < 1:
        raise ValueError("n_samples must be positive.")

    if jitter < 0:
        raise ValueError("jitter must be non-negative.")

    if rng is None:
        rng = np.random.default_rng()

    n_points = K.shape[0]

    # get square root of K via Kholesky decomposition with positive jitter added for numerical stability:
    L = np.linalg.cholesky(K + jitter * np.eye(n_points))

    # Draw independent standard normal samples
    z = rng.standard_normal((n_points, n_samples))

    return mu[:, None] + L @ z