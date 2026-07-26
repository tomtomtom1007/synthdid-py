"""Check the Frank-Wolfe solver against an independent convex solver.

The R package validates its solver against CVXR.  Here the reference is
accelerated projected gradient descent (FISTA) with an exact Euclidean
projection onto the simplex -- a different algorithm family from the
conditional-gradient method under test, so agreement is real evidence rather
than self-consistency.  Every reference solution is checked for optimality via
its Frank-Wolfe duality gap before being used.
"""

import numpy as np
import pytest

from synthdid import did_estimate, sc_estimate, synthdid_estimate

# Matrix products use ``.dot`` rather than ``@`` here for the same reason as in
# the package itself: ``@`` raises spurious RuntimeWarnings on macOS builds of
# numpy 2.0 (see the note in src/synthdid/solver.py), and pytest turns those
# into errors. Pure 1-D inner products are unaffected.

TOLERANCE = 0.03  # relative, matching the R package's own reference test


def project_simplex(v):
    """Euclidean projection onto ``{x >= 0, sum(x) == 1}`` (Duchi et al. 2008)."""
    u = np.sort(v)[::-1]
    cumulative = np.cumsum(u)
    support = u * np.arange(1, v.size + 1) > (cumulative - 1)
    rho = np.flatnonzero(support)[-1]
    theta = (cumulative[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0.0)


def simplex_least_squares(A, b, zeta=0.0, intercept=False, max_iter=200_000, tol=1e-14):
    """Minimize ``||Ax - b||^2 + zeta^2 n ||x||^2`` over the unit simplex."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n, p = A.shape
    if intercept:  # profile out the intercept by centering
        A = A - A.mean(axis=0, keepdims=True)
        b = b - b.mean()
    penalty = zeta ** 2 * n

    def objective(x):
        r = A.dot(x) - b
        return r @ r + penalty * (x @ x)

    def gradient(x):
        return 2 * A.T.dot(A.dot(x) - b) + 2 * penalty * x

    step = 1.0 / (2 * (np.linalg.norm(A, 2) ** 2 + penalty))
    x = np.full(p, 1.0 / p)
    y = x.copy()
    t = 1.0
    for _ in range(max_iter):
        x_next = project_simplex(y - step * gradient(y))
        t_next = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = x_next + ((t - 1) / t_next) * (x_next - x)
        if np.max(np.abs(x_next - x)) < tol:
            x = x_next
            break
        x, t = x_next, t_next

    # Frank-Wolfe duality gap, which upper-bounds suboptimality and is zero
    # exactly at the optimum. The synthetic-control problem is only
    # infinitesimally regularized, so we accept a small relative gap.
    grad = gradient(x)
    gap = grad @ x - grad.min()
    scale = max(abs(objective(x)), 1.0)
    assert gap / scale < 1e-4, f"reference solver did not converge (gap {gap:.3e})"
    return x


def noise_level(Y, N0, T0):
    return float(np.std(np.diff(Y[:N0, :T0], axis=1), ddof=1))


def synthdid_reference(Y, N0, T0):
    N, T = Y.shape
    sigma = noise_level(Y, N0, T0)
    lam = simplex_least_squares(
        Y[:N0, :T0], Y[:N0, T0:].mean(axis=1), zeta=1e-6 * sigma, intercept=True
    )
    omega = simplex_least_squares(
        Y[:N0, :T0].T,
        Y[N0:, :T0].mean(axis=0),
        zeta=((N - N0) * (T - T0)) ** 0.25 * sigma,
        intercept=True,
    )
    unit = np.concatenate([-omega, np.full(N - N0, 1 / (N - N0))])
    time = np.concatenate([-lam, np.full(T - T0, 1 / (T - T0))])
    return float(unit.dot(Y).dot(time))


def sc_reference(Y, N0, T0):
    N, T = Y.shape
    omega = simplex_least_squares(
        Y[:N0, :T0].T,
        Y[N0:, :T0].mean(axis=0),
        zeta=1e-6 * noise_level(Y, N0, T0),
        intercept=False,
    )
    unit = np.concatenate([-omega, np.full(N - N0, 1 / (N - N0))])
    time = np.concatenate([np.zeros(T0), np.full(T - T0, 1 / (T - T0))])
    return float(unit.dot(Y).dot(time))


def did_reference(Y, N0, T0):
    N, T = Y.shape
    unit = np.concatenate([np.full(N0, -1 / N0), np.full(N - N0, 1 / (N - N0))])
    time = np.concatenate([np.full(T0, -1 / T0), np.full(T - T0, 1 / (T - T0))])
    return float(unit.dot(Y).dot(time))


ACCURATE = dict(min_decrease=1e-6, max_iter=100_000)


def test_synthdid_agrees_with_a_convex_solver(prop99):
    ours = synthdid_estimate(prop99.Y, prop99.N0, prop99.T0, **ACCURATE)
    theirs = synthdid_reference(prop99.Y, prop99.N0, prop99.T0)
    assert float(ours) == pytest.approx(theirs, rel=TOLERANCE)


def test_sc_agrees_with_a_convex_solver(prop99):
    ours = sc_estimate(prop99.Y, prop99.N0, prop99.T0, **ACCURATE)
    theirs = sc_reference(prop99.Y, prop99.N0, prop99.T0)
    assert float(ours) == pytest.approx(theirs, rel=TOLERANCE)


def test_did_agrees_with_the_closed_form(prop99):
    ours = did_estimate(prop99.Y, prop99.N0, prop99.T0)
    assert float(ours) == pytest.approx(did_reference(prop99.Y, prop99.N0, prop99.T0), rel=1e-12)


def test_solver_reaches_a_comparable_objective(prop99):
    """Beyond the estimate, the weights themselves should be near-optimal."""
    Y, N0, T0 = prop99.Y, prop99.N0, prop99.T0
    sigma = noise_level(Y, N0, T0)
    zeta = ((Y.shape[0] - N0) * (Y.shape[1] - T0)) ** 0.25 * sigma

    A = Y[:N0, :T0].T
    b = Y[N0:, :T0].mean(axis=0)
    Ac = A - A.mean(axis=0, keepdims=True)
    bc = b - b.mean()

    def objective(x):
        r = Ac.dot(x) - bc
        return r @ r + zeta ** 2 * len(bc) * (x @ x)

    ours = synthdid_estimate(Y, N0, T0, **ACCURATE).weights.omega
    theirs = simplex_least_squares(A, b, zeta=zeta, intercept=True)
    assert objective(ours) <= objective(theirs) * 1.01
