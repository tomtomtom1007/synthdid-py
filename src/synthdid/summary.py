"""Summaries of a fitted estimate: influential controls, periods and dimensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .estimate import SynthDIDEstimate

__all__ = ["synthdid_controls", "summary", "SynthDIDSummary", "format_estimate"]

Estimates = Union[SynthDIDEstimate, Sequence[SynthDIDEstimate], Mapping[str, SynthDIDEstimate]]


def _as_named_estimates(estimates: Estimates) -> dict[str, SynthDIDEstimate]:
    """Normalize a single estimate, a sequence or a mapping into a name -> estimate dict."""
    if isinstance(estimates, SynthDIDEstimate):
        return {"estimate 1": estimates}
    if isinstance(estimates, Mapping):
        return dict(estimates)
    return {f"estimate {i + 1}": est for i, est in enumerate(estimates)}


def synthdid_controls(
    estimates: Estimates,
    sort_by: Union[int, str] = 0,
    mass: float = 0.9,
    weight_type: str = "omega",
) -> Union[pd.Series, pd.DataFrame]:
    """Table of the controls (or pre-periods) that carry the weight, sorted by weight.

    The table is truncated so that, for every estimate, the units shown account
    for at least ``mass`` of the total weight.

    Parameters
    ----------
    estimates : estimate, sequence or mapping of estimates
    sort_by : int or str
        Which estimate to sort by; position or name.
    mass : float
        Weight mass to retain.
    weight_type : {'omega', 'lambda'}
        ``omega`` gives control units, ``lambda`` gives pre-treatment periods.

    Returns
    -------
    Series for a single estimate, DataFrame for several.
    """
    if weight_type not in ("omega", "lambda"):
        raise ValueError('weight_type must be "omega" or "lambda"')
    named = _as_named_estimates(estimates)
    single = isinstance(estimates, SynthDIDEstimate)

    columns = {
        name: np.asarray(est.weights[weight_type], dtype=float) for name, est in named.items()
    }
    first = next(iter(named.values()))
    labels = (
        first.setup.unit_labels()[: first.setup.N0]
        if weight_type == "omega"
        else first.setup.period_labels()[: first.setup.T0]
    )

    table = pd.DataFrame(columns, index=pd.Index(labels, name=None))
    key = list(named)[sort_by] if isinstance(sort_by, int) else sort_by
    table = table.sort_values(key, ascending=False, kind="stable")

    lengths = []
    for name in table.columns:
        cumulative = table[name].cumsum().to_numpy()
        reached = np.flatnonzero(cumulative >= mass)
        lengths.append(int(reached[0]) + 1 if reached.size else len(table))
    table = table.iloc[: max(lengths)]

    return table.iloc[:, 0].rename(None) if single else table


@dataclass
class SynthDIDSummary:
    """Result of :func:`summary`."""

    estimate: float
    se: float
    controls: pd.Series
    periods: pd.Series
    dimensions: dict

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        d = self.dimensions
        lines = [
            f"synthdid: {self.estimate:.4f}  (se {self.se:.4f})",
            f"  N1={d['N1']}  N0={d['N0']}  effective N0={d['N0_effective']}",
            f"  T1={d['T1']}  T0={d['T0']}  effective T0={d['T0_effective']}",
            "",
            "Top control units (omega):",
            self.controls.to_string(),
            "",
            "Top pre-treatment periods (lambda):",
            self.periods.to_string(),
        ]
        return "\n".join(lines)


def summary(
    estimate: SynthDIDEstimate,
    weight_digits: int = 3,
    fast: bool = False,
    se_method: Optional[str] = None,
    replications: int = 200,
    random_state=None,
) -> SynthDIDSummary:
    """Estimate, standard error, effective sample sizes and the leading weights.

    Parameters
    ----------
    estimate : SynthDIDEstimate
    weight_digits : int
        Rounding applied to the displayed weights.
    fast : bool
        Use the jackknife rather than the bootstrap for the standard error.
    se_method : str, optional
        Overrides ``fast`` with an explicit method name.
    replications, random_state :
        Passed to the variance estimator.
    """
    from .vcov import vcov as _vcov

    setup = estimate.setup
    method = se_method or ("jackknife" if fast else "bootstrap")
    kwargs = {} if method == "jackknife" else {"replications": replications, "random_state": random_state}
    se = float(np.sqrt(_vcov(estimate, method=method, **kwargs).item()))

    omega = np.asarray(estimate.weights.omega, dtype=float)
    lam = np.asarray(estimate.weights.lambda_, dtype=float)
    dimensions = {
        "N1": setup.N1,
        "N0": setup.N0,
        "N0_effective": round(float(1.0 / np.sum(omega ** 2)), weight_digits)
        if np.any(omega)
        else float("inf"),
        "T1": setup.T1,
        "T0": setup.T0,
        "T0_effective": round(float(1.0 / np.sum(lam ** 2)), weight_digits)
        if np.any(lam)
        else float("inf"),
    }
    return SynthDIDSummary(
        estimate=float(estimate),
        se=se,
        controls=synthdid_controls(estimate, weight_type="omega").round(weight_digits),
        periods=synthdid_controls(estimate, weight_type="lambda").round(weight_digits),
        dimensions=dimensions,
    )


def format_estimate(estimate: SynthDIDEstimate, se_method: str = "jackknife") -> str:
    """One-line summary, mirroring ``format.synthdid_estimate`` in the R package."""
    info = summary(estimate, fast=True, se_method=se_method)
    d = info.dimensions
    n0_eff, t0_eff = d["N0_effective"], d["T0_effective"]
    return (
        f"{estimate.estimator}: {float(estimate):.3f} +- {1.96 * info.se:.3f}. "
        f"Effective N0/N0 = {n0_eff:.1f}/{d['N0']}~{n0_eff / d['N0']:.1f}. "
        f"Effective T0/T0 = {t0_eff:.1f}/{d['T0']}~{t0_eff / d['T0']:.1f}. "
        f"N1,T1 = {d['N1']},{d['T1']}."
    )
