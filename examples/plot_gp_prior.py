r"""Visualize samples from a Gaussian Process prior.

This example demonstrates how to draw and plot sample functions
from a zero-mean Gaussian Process (GP) with an RBF covariance kernel.

The Gaussian Process is defined as:

\f[
    f(x) \sim \mathcal{GP}(0, k(x, x'))
\f]

For a finite set of input points
\f$ X = \{x_1, \ldots, x_n\} \f$, the corresponding function values
follow a multivariate Gaussian distribution:

\f[
    \mathbf{f} \sim \mathcal{N}(\mathbf{0}, K)
\f]

where \f$ K_{ij} = k(x_i, x_j) \f$.

The script generates ten sample functions on 50 equally spaced
input points in the interval [-5, 5] and visualizes them
using Matplotlib.
"""
import matplotlib.pyplot as plt
import numpy as np
from gaussian_processes.kernels import rbf_kernel
from gaussian_processes.prior import sample_from_multivariate_gaussian


def main() -> None:
    r"""Generate and plot sample functions from a Gaussian Process prior."""
    x = np.linspace(-5,5,50)
    x = x[:,None]  # shape (n_points,1)

    K = rbf_kernel(x,x)

    mu = np.zeros(x.shape[0])
    mu = np.ones(x.shape[0])*5

    f_prior_samples = sample_from_multivariate_gaussian(mu,K)

    fig, ax = plt.subplots()

    ax.plot(x[:, 0], f_prior_samples)

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Gaussian Process Prior")
    plt.show()


if __name__ == "__main__":
    main()