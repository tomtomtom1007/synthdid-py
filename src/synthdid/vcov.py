"""Variance estimation: the bootstrap, jackknife and placebo of Arkhangelsky et al.

Port of ``R/vcov.R``.  The three algorithms correspond to Algorithms 2, 3 and 4
of the paper.  The jackknife is fast but is not recommended for synthetic
control estimates; the placebo is the only option that works with a single
treated unit.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .estimate import SynthDIDEstimate, Weights, refit
from .utils import sum_normalize

__all__ = [
    "vcov",
    "FITTED_WEIGHTS",
    "synthdid_se",
    "bootstrap_se",
    "bootstrap_sample",
    "jackknife_se",
    "placebo_se",
]

RandomState = Union[None, int, np.random.Generator]

_METHODS = ("bootstrap", "jackknife", "placebo")

#: Sentinel meaning "use the estimate's own weights", so that an explicit
#: ``weights=None`` can request the ordinary (re-solving) jackknife instead.
FITTED_WEIGHTS = object()


def vcov(
    estimate: SynthDIDEstimate,
    method: str = "bootstrap",
    replications: int = 200,
    random_state: RandomState = None,
) -> np.ndarray:
    """Variance of a synthdid estimate, as a 1x1 matrix.

    Parameters
    ----------
    estimate : SynthDIDEstimate
    method : {'bootstrap', 'jackknife', 'placebo'}
        ``bootstrap`` (Algorithm 2) is the default and the most generally
        reliable; ``jackknife`` (Algorithm 3) is much faster but is not
        recommended for synthetic control and returns ``nan`` with one treated
        unit; ``placebo`` (Algorithm 4) is the option to use when there is a
        single treated unit.
    replications : int
        Number of bootstrap or placebo draws.  Ignored by the jackknife.
    random_state : int or Generator, optional
        Seed for reproducible resampling.

    Returns
    -------
    ndarray of shape (1, 1)
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if method == "bootstrap":
        se = bootstrap_se(estimate, replications, random_state=random_state)
    elif method == "jackknife":
        se = jackknife_se(estimate)
    else:
        se = placebo_se(estimate, replications, random_state=random_state)
    return np.array([[se ** 2]])


def synthdid_se(estimate: SynthDIDEstimate, **kwargs) -> float:
    """Standard error of a synthdid estimate; a thin wrapper around :func:`vcov`."""
    return float(np.sqrt(vcov(estimate, **kwargs).item()))


def _weights_for_subset(weights: Weights, control_index: np.ndarray) -> Weights:
    """Copy ``weights`` with ``omega`` restricted to ``control_index`` and renormalized."""
    out = weights.copy()
    out.omega = sum_normalize(weights.omega[control_index])
    out.vals = None
    out.lambda_vals = None
    out.omega_vals = None
    return out


def bootstrap_sample(
    estimate: SynthDIDEstimate,
    replications: int,
    random_state: RandomState = None,
) -> np.ndarray:
    """Draw ``replications`` bootstrap estimates by resampling units with replacement.

    Draws that contain only treated or only control units are discarded and
    redrawn.  Returns ``[nan]`` when there is a single treated unit, for which
    the bootstrap is not defined.
    """
    rng = np.random.default_rng(random_state)
    setup = estimate.setup
    N = setup.Y.shape[0]
    if setup.N0 == N - 1:
        return np.array([np.nan])

    estimates = np.empty(replications)
    count = 0
    guard = 0
    max_draws = max(1000, 100 * replications)
    while count < replications:
        guard += 1
        if guard > max_draws:
            raise RuntimeError(
                "bootstrap failed to draw samples containing both treated and "
                "control units; the panel may be too small for this method."
            )
        ind = np.sort(rng.integers(0, N, size=N))
        n0_boot = int((ind < setup.N0).sum())
        if n0_boot == 0 or n0_boot == N:
            continue
        weights_boot = _weights_for_subset(estimate.weights, ind[ind < setup.N0])
        estimates[count] = refit(
            estimate,
            Y=setup.Y[ind],
            N0=n0_boot,
            T0=setup.T0,
            X=setup.X[ind],
            weights=weights_boot,
        )
        count += 1
    return estimates


def bootstrap_se(
    estimate: SynthDIDEstimate,
    replications: int = 200,
    random_state: RandomState = None,
) -> float:
    """Bootstrap standard error (Algorithm 2 of Arkhangelsky et al.)."""
    draws = bootstrap_sample(estimate, replications, random_state=random_state)
    if draws.size < 2 or np.any(np.isnan(draws)):
        return float("nan")
    return float(
        np.sqrt((replications - 1) / replications) * np.std(draws, ddof=1)
    )


def jackknife_se(
    estimate: SynthDIDEstimate,
    weights: Union[Weights, None, object] = FITTED_WEIGHTS,
) -> float:
    """Fixed-weights jackknife standard error (Algorithm 3 of Arkhangelsky et al.).

    Pass ``weights=None`` explicitly for the ordinary jackknife, which re-solves
    for the weights in every leave-one-out replicate.  Returns ``nan`` when
    there is a single treated unit, or -- for the fixed-weights version -- when
    exactly one control carries nonzero weight.
    """
    if weights is FITTED_WEIGHTS:
        weights = estimate.weights

    setup = estimate.setup
    N = setup.Y.shape[0]
    opts = dict(estimate.opts)
    if weights is not None:
        opts["update_omega"] = False
        opts["update_lambda"] = False

    if setup.N0 == N - 1:
        return float("nan")
    if weights is not None and int(np.count_nonzero(weights.omega)) == 1:
        return float("nan")

    base = SynthDIDEstimate(
        float(estimate), estimator=estimate.estimator, weights=estimate.weights,
        setup=setup, opts=opts,
    )

    u = np.empty(N)
    all_rows = np.arange(N)
    for i in range(N):
        ind = np.delete(all_rows, i)
        weights_jk = (
            None if weights is None else _weights_for_subset(weights, ind[ind < setup.N0])
        )
        u[i] = refit(
            base,
            Y=setup.Y[ind],
            N0=int((ind < setup.N0).sum()),
            T0=setup.T0,
            X=setup.X[ind],
            weights=weights_jk,
        )
    return float(np.sqrt(((N - 1) / N) * (N - 1) * np.var(u, ddof=1)))


def placebo_se(
    estimate: SynthDIDEstimate,
    replications: int = 200,
    random_state: RandomState = None,
) -> float:
    """Placebo standard error (Algorithm 4 of Arkhangelsky et al.).

    Repeatedly relabels ``N1`` of the control units as treated and re-estimates.
    Requires strictly more control units than treated units.
    """
    rng = np.random.default_rng(random_state)
    setup = estimate.setup
    N1 = setup.N1
    if setup.N0 <= N1:
        raise ValueError(
            "must have more controls than treated units to use the placebo se"
        )

    draws = np.empty(replications)
    n0 = setup.N0 - N1
    for b in range(replications):
        ind = rng.permutation(setup.N0)
        weights_boot = _weights_for_subset(estimate.weights, ind[:n0])
        draws[b] = refit(
            estimate,
            Y=setup.Y[ind],
            N0=n0,
            T0=setup.T0,
            X=setup.X[ind],
            weights=weights_boot,
        )
    return float(np.sqrt((replications - 1) / replications) * np.std(draws, ddof=1))
