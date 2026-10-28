"""Thin wrapper around :mod:`qpsolvers` for the QPs in this package.

The CBF safety-filter QP in :mod:`gaussian_processes.controllers` is
solved with OSQP via :func:`qpsolvers.solve_qp`.
"""
from typing import Optional

import numpy as np
import qpsolvers


def solve_qp(
    G: np.ndarray,
    q: np.ndarray,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    solver: Optional[str] = None,
    **solver_kwargs,
) -> Optional[np.ndarray]:
    """Solve the convex QP

    .. math::

        \\min_x \\; \\tfrac{1}{2}\\, x^T G x + q^T x
        \\quad \\text{s.t.} \\quad A x \\le b

    using :func:`qpsolvers.solve_qp`.  With ``solver=None`` the first
    installed backend is used (here: OSQP).

    Parameters
    ----------
    G:
        (n, n) symmetric positive (semi)definite Hessian.
    q:
        (n,) linear cost term.
    A, b:
        Linear inequality constraints ``A @ x <= b`` with shapes
        (m, n) and (m,).  May be omitted for an unconstrained problem.
    solver:
        Backend name passed to :mod:`qpsolvers` (``None`` = auto).
    **solver_kwargs:
        Extra options forwarded to the backend (e.g. ``eps_abs``).

    Returns
    -------
    (n,) solution ``x``, or ``None`` if the problem is infeasible or
    the solver fails.  ADMM-type backends (OSQP) are iterative, so the
    returned point may violate the constraints by a small tolerance.
    """
    q = np.asarray(q, dtype=float)
    n = int(q.shape[0])
    G = np.atleast_2d(np.asarray(G, dtype=float))
    if A is None:
        A = np.zeros((0, n))
        b = np.zeros(0)
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float)
    if solver is None:
        solver = qpsolvers.available_solvers[0]
    if solver == "osqp":
        # OSQP's defaults leave a few 1e-4 of constraint slack; the
        # CBF safety filter wants the barrier row tight.
        solver_kwargs.setdefault("eps_abs", 1e-10)
        solver_kwargs.setdefault("eps_rel", 1e-10)
    return qpsolvers.solve_qp(
        P=G, q=q, G=A, h=b, solver=solver, **solver_kwargs
    )
