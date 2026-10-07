r"""Kernel functions for Gaussian processes."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def rbf_kernel(x1: ArrayLike,
            x2: ArrayLike,
            *,
            length_scale: float = 1.0,
            variance: float = 1.0,
            )-> NDArray[np.float64]:
    r"""Compute the radial basis function (RBF) kernel matrix.

    The RBF kernel, also known as the squared exponential kernel, is

    \f[
        k(\mathbf{x}, \mathbf{x}')
        =
        \sigma_f^2
        \exp\left(
            -\frac{
                \lVert \mathbf{x} - \mathbf{x}' \rVert^2
            }{
                2\ell^2
            }
        \right),
    \f]

    where \f$\ell\f$ is the length scale and
    \f$\sigma_f^2\f$ is the signal variance.

    @param x1
        First set of samples with shape
        `(n_samples_1, n_features)`.
    @param x2
        Second set of samples with shape
        `(n_samples_2, n_features)`.
    @param length_scale
        Positive kernel length scale \f$\ell\f$.
    @param variance
        Positive signal variance \f$\sigma_f^2\f$.

    @return
        Kernel matrix with shape
        `(n_samples_1, n_samples_2)`.

    @throws ValueError
        If an input is not two-dimensional, if the numbers of features
        differ, or if a kernel parameter is not positive.
    """
    if length_scale <= 0.0:
            raise ValueError("length_scale must be positive")

    if variance <= 0.0:
        raise ValueError("variance must be positive")

    x1_array = np.asarray(x1, dtype=np.float64)
    x2_array = np.asarray(x2, dtype=np.float64)

    if x1_array.ndim != 2 or x2_array.ndim != 2:
        raise ValueError("inputs must be two-dimensional")

    if x1_array.shape[1] != x2_array.shape[1]:
        raise ValueError("inputs must have the same number of features")

    diff = (
        x1_array[:, None, :]   # (n_a, 1, d)
        - x2_array[None, :, :]   # (1, n_b, d)
    )
    sq_dists = np.sum(diff**2,axis=-1)

    return variance* np.exp(-sq_dists / (2.0 * length_scale**2))
