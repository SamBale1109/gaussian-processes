import numpy as np
import pytest

from gaussian_processes.kernels import rbf_kernel


def test_rbf_kernel_shape() -> None:
    x1 = np.zeros((3, 2))
    x2 = np.zeros((5, 2))

    result = rbf_kernel(x1, x2)

    assert result.shape == (3, 5)


def test_rbf_kernel_is_symmetric() -> None:
    x = np.array([[0.0], [1.0], [2.0]])

    result = rbf_kernel(x, x)

    np.testing.assert_allclose(result, result.T)


def test_rbf_kernel_diagonal_equals_variance() -> None:
    x = np.array([[0.0], [1.0], [2.0]])

    result = rbf_kernel(x, x, variance=2.5)

    np.testing.assert_allclose(np.diag(result), 2.5)


def test_rbf_kernel_rejects_non_positive_length_scale() -> None:
    x = np.array([[0.0]])

    with pytest.raises(ValueError):
        rbf_kernel(x, x, length_scale=0.0)
