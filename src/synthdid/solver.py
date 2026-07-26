"""Frank-Wolfe solvers for synthetic difference-in-differences weights.

This is a line-by-line port of ``R/solver.R`` from the reference R package
`synthdid <https://github.com/synth-inference/synthdid>`_ by Arkhangelsky,
Athey, Hirshberg, Imbens and Wager.  The numerical behaviour -- including the
stopping rules, the exact line search and the order in which weights are
updated -- is preserved so that estimates agree with the R implementation to
solver tolerance.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "contract3",
    "fw_step",
    "sc_weight_fw",
    "sc_weight_fw_covariates",
    "sparsify_function",
]


def contract3(X: np.ndarray, v: Optional[np.ndarray]) -> np.ndarray:
    """Contract the third axis of ``X`` (N x T x C) against ``v`` (length C).

    Returns an N x T matrix of zeros when there are no covariates.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"X must be a 3-d array, got shape {X.shape}")
    v = np.zeros(0) if v is None else np.asarray(v, dtype=float).ravel()
    if X.shape[2] != v.size:
        raise ValueError(f"X has {X.shape[2]} covariates but v has length {v.size}")
    if v.size == 0:
        return np.zeros(X.shape[:2])
    return np.tensordot(X, v, axes=([2], [0]))


def fw_step(
    A: np.ndarray,
    x: np.ndarray,
    b: np.ndarray,
    eta: float,
    alpha: Optional[float] = None,
) -> np.ndarray:
    """One Frank-Wolfe step for ``||Ax - b||^2 + eta * ||x||^2`` over the simplex.

    With ``alpha=None`` (the default) an exact line search is used; otherwise the
    step size is fixed at ``alpha``.
    """
    Ax = A @ x
    half_grad = (Ax - b) @ A + eta * x
    i = int(np.argmin(half_grad))

    if alpha is not None:
        x = x * (1.0 - alpha)
        x[i] += alpha
        return x

    d_x = -x.copy()
    d_x[i] = 1.0 - x[i]
    if not np.any(d_x):
        return x

    d_err = A[:, i] - Ax
    denominator = d_err @ d_err + eta * (d_x @ d_x)
    if denominator <= 0:
        # The objective is flat along d_x; no improvement is possible.
        return x
    step = -(half_grad @ d_x) / denominator
    constrained_step = min(1.0, max(0.0, float(step)))
    return x + constrained_step * d_x


def sparsify_function(v: np.ndarray) -> np.ndarray:
    """Zero out weights no larger than a quarter of the largest, then renormalize."""
    v = np.asarray(v, dtype=float).copy()
    if v.size == 0:
        return v
    v[v <= v.max() / 4.0] = 0.0
    total = v.sum()
    if total == 0:
        return np.full(v.size, 1.0 / v.size)
    return v / total


def sc_weight_fw(
    Y: np.ndarray,
    zeta: float,
    intercept: bool = True,
    lambda_: Optional[np.ndarray] = None,
    min_decrease: float = 1e-3,
    max_iter: int = 1000,
) -> dict:
    """Frank-Wolfe solver for synthetic-control weights with exact line search.

    Solves for weights on the first ``T0 = ncol(Y) - 1`` columns of ``Y`` that
    best predict the final column, penalized by ``zeta^2 * ||w||^2``.

    Parameters
    ----------
    Y : array, shape (N0, T0 + 1)
        The last column is the target.
    zeta : float
        Ridge regularization level.
    intercept : bool
        If True, allow an intercept by centering each column of ``Y``.
    lambda_ : array, optional
        Initial weights.  Defaults to uniform.
    min_decrease, max_iter :
        Stop once an iteration decreases the objective by less than
        ``min_decrease ** 2``, or after ``max_iter`` iterations.

    Returns
    -------
    dict with keys ``lambda`` (the weights) and ``vals`` (objective per iteration).
    """
    Y = np.asarray(Y, dtype=float)
    T0 = Y.shape[1] - 1
    N0 = Y.shape[0]
    if lambda_ is None:
        lambda_ = np.full(T0, 1.0 / T0) if T0 > 0 else np.zeros(0)
    else:
        lambda_ = np.asarray(lambda_, dtype=float).copy()
    if lambda_.size != T0:
        raise ValueError(f"initial weights have length {lambda_.size}, expected {T0}")

    if intercept:
        Y = Y - Y.mean(axis=0, keepdims=True)

    A = Y[:, :T0]
    b = Y[:, T0]
    eta = N0 * zeta ** 2
    coef = np.empty(T0 + 1)
    coef[T0] = -1.0

    vals: list[float] = []
    while len(vals) < max_iter and (
        len(vals) < 2 or vals[-2] - vals[-1] > min_decrease ** 2
    ):
        lambda_ = fw_step(A, lambda_, b, eta)
        coef[:T0] = lambda_
        err = Y @ coef
        vals.append(zeta ** 2 * (lambda_ @ lambda_) + (err @ err) / N0)

    return {"lambda": lambda_, "vals": np.asarray(vals)}


def sc_weight_fw_covariates(
    Y: np.ndarray,
    X: Optional[np.ndarray] = None,
    zeta_lambda: float = 0.0,
    zeta_omega: float = 0.0,
    lambda_intercept: bool = True,
    omega_intercept: bool = True,
    min_decrease: float = 1e-3,
    max_iter: int = 1000,
    lambda_: Optional[np.ndarray] = None,
    omega: Optional[np.ndarray] = None,
    beta: Optional[np.ndarray] = None,
    update_lambda: bool = True,
    update_omega: bool = True,
) -> dict:
    """Joint solver for ``lambda``, ``omega`` and covariate coefficients ``beta``.

    Alternates exact-line-search Frank-Wolfe steps for the weights with
    ``1/t``-sized gradient steps for ``beta``, as in ``sc.weight.fw.covariates``
    in the R package.

    ``Y`` is the collapsed (N0+1) x (T0+1) outcome matrix and ``X`` the
    correspondingly collapsed (N0+1) x (T0+1) x C covariate array.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be a matrix")
    if X is None:
        X = np.zeros(Y.shape + (0,))
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[:, :, None]
    if X.shape[:2] != Y.shape:
        raise ValueError("X and Y must agree on their first two dimensions")
    if not np.all(np.isfinite(Y)) or not np.all(np.isfinite(X)):
        raise ValueError("Y and X must be finite")

    T0 = Y.shape[1] - 1
    N0 = Y.shape[0] - 1
    n_cov = X.shape[2]

    lambda_ = np.full(T0, 1.0 / T0) if lambda_ is None else np.asarray(lambda_, float).copy()
    omega = np.full(N0, 1.0 / N0) if omega is None else np.asarray(omega, float).copy()
    beta = np.zeros(n_cov) if beta is None else np.asarray(beta, float).copy()

    lambda_coef = np.empty(T0 + 1)
    lambda_coef[T0] = -1.0
    omega_coef = np.empty(N0 + 1)
    omega_coef[N0] = -1.0

    def update_weights(Y_beta: np.ndarray, lambda_: np.ndarray, omega: np.ndarray) -> dict:
        Y_lambda = Y_beta[:N0, :]
        if lambda_intercept:
            Y_lambda = Y_lambda - Y_lambda.mean(axis=0, keepdims=True)
        if update_lambda:
            lambda_ = fw_step(
                Y_lambda[:, :T0], lambda_, Y_lambda[:, T0], N0 * zeta_lambda ** 2
            )
        lambda_coef[:T0] = lambda_
        err_lambda = Y_lambda @ lambda_coef

        Y_omega = Y_beta[:, :T0].T
        if omega_intercept:
            Y_omega = Y_omega - Y_omega.mean(axis=0, keepdims=True)
        if update_omega:
            omega = fw_step(
                Y_omega[:, :N0], omega, Y_omega[:, N0], T0 * zeta_omega ** 2
            )
        omega_coef[:N0] = omega
        err_omega = Y_omega @ omega_coef

        val = (
            zeta_omega ** 2 * (omega @ omega)
            + zeta_lambda ** 2 * (lambda_ @ lambda_)
            + (err_omega @ err_omega) / T0
            + (err_lambda @ err_lambda) / N0
        )
        return {
            "val": val,
            "lambda": lambda_,
            "omega": omega,
            "err_lambda": err_lambda,
            "err_omega": err_omega,
        }

    Y_beta = Y - contract3(X, beta)
    weights = update_weights(Y_beta, lambda_, omega)

    vals: list[float] = []
    while len(vals) < max_iter and (
        len(vals) < 2 or abs(vals[-2] - vals[-1]) > min_decrease ** 2
    ):
        t = len(vals) + 1
        if n_cov == 0:
            grad_beta = np.zeros(0)
        else:
            lam_coef = np.append(weights["lambda"], -1.0)
            om_coef = np.append(weights["omega"], -1.0)
            grad_beta = -np.array(
                [
                    weights["err_lambda"] @ X[:N0, :, c] @ lam_coef / N0
                    + weights["err_omega"] @ X[:, :T0, c].T @ om_coef / T0
                    for c in range(n_cov)
                ]
            )
        beta = beta - grad_beta / t
        Y_beta = Y - contract3(X, beta)
        weights = update_weights(Y_beta, weights["lambda"], weights["omega"])
        vals.append(weights["val"])

    return {
        "lambda": weights["lambda"],
        "omega": weights["omega"],
        "beta": beta,
        "vals": np.asarray(vals),
    }
