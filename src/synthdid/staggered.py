"""Staggered adoption: units that start treatment at different times.

The reference R package only handles simultaneous adoption.  This module lifts
that restriction the way Clarke, Pailanir, Athey and Imbens do in the Stata
``sdid`` package: split the panel into adoption cohorts, run the block
estimator on each, and combine the cohort estimates as

.. math:: \\hat\\tau = \\sum_a \\frac{N^a_{tr} T^a_{post}}{\\sum_b N^b_{tr} T^b_{post}} \\hat\\tau^a

so each cohort is weighted by the number of treated unit-periods it
contributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .estimate import SynthDIDEstimate, did_estimate, sc_estimate, synthdid_estimate
from .utils import ColumnRef, _resolve_column, sum_normalize

__all__ = [
    "StaggeredPanel",
    "StaggeredEstimate",
    "staggered_panel_matrices",
    "staggered_synthdid_estimate",
]

_ESTIMATORS: dict[str, Callable[..., SynthDIDEstimate]] = {
    "synthdid": synthdid_estimate,
    "sc": sc_estimate,
    "did": did_estimate,
}

RandomState = Union[None, int, np.random.Generator]


@dataclass
class StaggeredPanel:
    """A balanced panel in which treatment adoption may differ across units.

    Attributes
    ----------
    Y : ndarray (N, T)
        Outcomes, rows sorted by adoption time with never-treated units first.
    adoption : ndarray (N,)
        Column index at which each unit adopts treatment; ``-1`` for units that
        are never treated.
    units, time : ndarray
        Row and column labels.
    X : ndarray (N, T, C), optional
        Time-varying covariates.
    """

    Y: np.ndarray
    adoption: np.ndarray
    units: np.ndarray
    time: np.ndarray
    X: Optional[np.ndarray] = None

    @property
    def never_treated(self) -> np.ndarray:
        return self.adoption < 0

    @property
    def adoption_times(self) -> np.ndarray:
        """Distinct adoption column indices, in order."""
        return np.unique(self.adoption[self.adoption >= 0])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        cohorts = self.adoption_times
        return (
            f"StaggeredPanel(N={self.Y.shape[0]} units "
            f"[{int(self.never_treated.sum())} never treated], "
            f"T={self.Y.shape[1]} periods, {len(cohorts)} adoption cohorts: "
            f"{list(self.time[cohorts])})"
        )


def staggered_panel_matrices(
    panel: pd.DataFrame,
    unit: ColumnRef = 1,
    time: ColumnRef = 2,
    outcome: ColumnRef = 3,
    treatment: ColumnRef = 4,
    covariates: Optional[Sequence[ColumnRef]] = None,
) -> StaggeredPanel:
    """Reshape a long panel with staggered adoption into matrix form.

    Treatment must be absorbing: once a unit is treated it stays treated.  The
    panel must be balanced.

    Raises
    ------
    ValueError
        If the panel is unbalanced, has missing values, or treatment switches
        off again for some unit.
    """
    if not isinstance(panel, pd.DataFrame):
        raise ValueError("Unsupported input type `panel`: expected a pandas DataFrame.")

    unit_c = _resolve_column(panel, unit, "unit")
    time_c = _resolve_column(panel, time, "time")
    outcome_c = _resolve_column(panel, outcome, "outcome")
    treatment_c = _resolve_column(panel, treatment, "treatment")
    cov_c = [_resolve_column(panel, c, "covariate") for c in (covariates or [])]

    data = panel.loc[:, [unit_c, time_c, outcome_c, treatment_c, *cov_c]].copy()
    if data.isna().to_numpy().any():
        raise ValueError("Missing values in `panel`.")

    treat = data[treatment_c]
    treat = treat.astype(int) if treat.dtype == bool else pd.to_numeric(treat, errors="coerce")
    if treat.isna().any() or not treat.isin([0, 1]).all():
        raise ValueError("The treatment status should be in 0 or 1.")
    if treat.nunique() == 1:
        raise ValueError("There is no variation in treatment status.")
    data[treatment_c] = treat.astype(int)

    counts = data.groupby([unit_c, time_c], observed=True, sort=False).size()
    if len(counts) != data[unit_c].nunique() * data[time_c].nunique() or not (counts == 1).all():
        raise ValueError(
            "Input `panel` must be a balanced panel: it must have an observation "
            "for every unit at every time."
        )

    def wide(col: str, dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frame = (
            data.pivot(index=unit_c, columns=time_c, values=col)
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        return frame.to_numpy(dtype=dtype), frame.index.to_numpy(), frame.columns.to_numpy()

    Y, units, times = wide(outcome_c, float)
    W, _, _ = wide(treatment_c, int)

    adoption = np.full(Y.shape[0], -1, dtype=int)
    for i, row in enumerate(W):
        treated_at = np.flatnonzero(row == 1)
        if treated_at.size == 0:
            continue
        start = int(treated_at[0])
        if not np.all(row[start:] == 1):
            raise ValueError(
                f"Treatment is not absorbing for unit {units[i]!r}: it switches "
                "off after being switched on."
            )
        if start == 0:
            raise ValueError(
                f"Unit {units[i]!r} is treated in the first period, so it has no "
                "pre-treatment data. Drop it or extend the panel."
            )
        adoption[i] = start

    # never-treated (adoption == -1) first, then cohorts in adoption order
    order = np.lexsort((np.arange(len(units)), adoption))

    X = None
    if cov_c:
        X = np.stack([wide(c, float)[0] for c in cov_c], axis=2)[order]

    return StaggeredPanel(
        Y=np.ascontiguousarray(Y[order]),
        adoption=adoption[order],
        units=units[order],
        time=times,
        X=X,
    )


@dataclass
class StaggeredEstimate:
    """Aggregate ATT across adoption cohorts, plus the per-cohort estimates."""

    att: float
    by_cohort: pd.DataFrame
    estimates: dict[Any, SynthDIDEstimate] = field(default_factory=dict)
    panel: Optional[StaggeredPanel] = None
    estimator: str = "synthdid"
    control_pool: str = "never_treated"

    def __float__(self) -> float:
        return float(self.att)

    def se(
        self,
        method: str = "bootstrap",
        replications: int = 200,
        random_state: RandomState = None,
    ) -> float:
        """Standard error of the aggregate ATT.

        Parameters
        ----------
        method : {'bootstrap', 'jackknife', 'placebo'}
            ``bootstrap`` resamples units with replacement and refits
            everything; ``jackknife`` leaves one unit out at a time holding the
            fitted weights fixed (fast); ``placebo`` relabels never-treated
            units as treated, reusing the observed adoption pattern.
        """
        return _staggered_se(self, method=method, replications=replications, random_state=random_state)

    def vcov(self, **kwargs) -> np.ndarray:
        return np.array([[self.se(**kwargs) ** 2]])

    def ci(self, level: float = 0.95, **kwargs) -> tuple[float, float]:
        from scipy.stats import norm

        se = self.se(**kwargs)
        z = norm.ppf(0.5 + level / 2.0)
        return (self.att - z * se, self.att + z * se)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        n_cohorts = len(self.by_cohort)
        return (
            f"staggered {self.estimator}: ATT = {self.att:.4f} "
            f"over {n_cohorts} adoption cohort(s)\n{self.by_cohort.to_string(index=False)}"
        )


def _cohort_blocks(panel: StaggeredPanel, control_pool: str) -> list[dict]:
    """Build the (rows, columns, N0, T0) block for each adoption cohort."""
    adoption = panel.adoption
    cohorts = panel.adoption_times
    T = panel.Y.shape[1]
    blocks = []
    for k, a in enumerate(cohorts):
        treated_rows = np.flatnonzero(adoption == a)
        if control_pool == "never_treated":
            end = T
            control_rows = np.flatnonzero(adoption < 0)
        elif control_pool == "not_yet_treated":
            later = cohorts[cohorts > a]
            end = int(later[0]) if later.size else T
            control_rows = np.flatnonzero((adoption < 0) | (adoption >= end))
        else:
            raise ValueError(
                'control_pool must be "never_treated" or "not_yet_treated", '
                f"got {control_pool!r}"
            )
        if control_rows.size == 0:
            raise ValueError(
                f"No control units available for the cohort adopting at "
                f"{panel.time[a]}. With control_pool={control_pool!r} every unit "
                "is treated by then."
            )
        blocks.append(
            {
                "adoption_index": int(a),
                "adoption_time": panel.time[a],
                "rows": np.concatenate([control_rows, treated_rows]),
                "n0": int(control_rows.size),
                "t0": int(a),
                "columns": np.arange(end),
                "n_treated": int(treated_rows.size),
                "n_post": int(end - a),
            }
        )
    return blocks


def staggered_synthdid_estimate(
    panel: Union[StaggeredPanel, pd.DataFrame],
    estimator: str = "synthdid",
    control_pool: str = "never_treated",
    unit: ColumnRef = 1,
    time: ColumnRef = 2,
    outcome: ColumnRef = 3,
    treatment: ColumnRef = 4,
    covariates: Optional[Sequence[ColumnRef]] = None,
    **kwargs,
) -> StaggeredEstimate:
    """Synthetic difference-in-differences under staggered adoption.

    Each adoption cohort is estimated as its own treated block against a pool of
    untreated controls, and the cohort ATTs are averaged with weights
    proportional to treated unit-periods.  With a single adoption date this
    reduces exactly to :func:`~synthdid.estimate.synthdid_estimate`.

    Parameters
    ----------
    panel : StaggeredPanel or DataFrame
        A long panel (reshaped for you) or an already-reshaped panel.
    estimator : {'synthdid', 'sc', 'did'}
        Block estimator applied within each cohort.
    control_pool : {'never_treated', 'not_yet_treated'}
        ``never_treated`` (the default, matching Stata's ``sdid``) uses only
        units that are never treated.  ``not_yet_treated`` also admits units
        that adopt later, truncating the cohort's post-period at the next
        adoption date so those controls stay untreated throughout.
    unit, time, outcome, treatment, covariates :
        Column identifiers, used only when ``panel`` is a DataFrame.
    **kwargs
        Passed through to the block estimator.

    Returns
    -------
    StaggeredEstimate

    Examples
    --------
    >>> est = staggered_synthdid_estimate(df, unit='state', time='year',   # doctest: +SKIP
    ...                                   outcome='y', treatment='d')
    >>> est.att, est.se(method='jackknife')                                # doctest: +SKIP
    """
    if isinstance(panel, pd.DataFrame):
        panel = staggered_panel_matrices(
            panel,
            unit=unit,
            time=time,
            outcome=outcome,
            treatment=treatment,
            covariates=covariates,
        )
    if estimator not in _ESTIMATORS:
        raise ValueError(f"estimator must be one of {sorted(_ESTIMATORS)}, got {estimator!r}")

    blocks = _cohort_blocks(panel, control_pool)
    fn = _ESTIMATORS[estimator]

    rows_out, estimates = [], {}
    for block in blocks:
        rows, cols = block["rows"], block["columns"]
        Y = panel.Y[np.ix_(rows, cols)]
        X = None if panel.X is None else panel.X[np.ix_(rows, cols, np.arange(panel.X.shape[2]))]
        est = fn(
            Y,
            block["n0"],
            block["t0"],
            X=X,
            unit_names=panel.units[rows],
            time_labels=panel.time[cols],
            **kwargs,
        )
        estimates[block["adoption_time"]] = est
        rows_out.append(
            {
                "adoption_time": block["adoption_time"],
                "n_treated": block["n_treated"],
                "n_control": block["n0"],
                "n_pre": block["t0"],
                "n_post": block["n_post"],
                "tau": float(est),
                "weight": float(block["n_treated"] * block["n_post"]),
            }
        )

    by_cohort = pd.DataFrame(rows_out)
    by_cohort["weight"] /= by_cohort["weight"].sum()
    att = float((by_cohort["weight"] * by_cohort["tau"]).sum())

    return StaggeredEstimate(
        att=att,
        by_cohort=by_cohort,
        estimates=estimates,
        panel=panel,
        estimator=estimator,
        control_pool=control_pool,
    )


def _att_with_fixed_weights(
    panel: StaggeredPanel,
    blocks: list[dict],
    fitted: Mapping[Any, SynthDIDEstimate],
    keep: np.ndarray,
) -> float:
    """Recompute the aggregate ATT on a subset of units, holding weights fixed."""
    taus, weights = [], []
    for block in blocks:
        rows = block["rows"]
        n0 = block["n0"]
        kept = keep[rows]
        control_kept = kept[:n0]
        treated_kept = kept[n0:]
        if not treated_kept.any() or not control_kept.any():
            continue
        est = fitted[block["adoption_time"]]
        omega = sum_normalize(np.asarray(est.weights.omega)[control_kept])
        lam = np.asarray(est.weights.lambda_)
        n_post = block["n_post"]
        n_treated = int(treated_kept.sum())

        Y = panel.Y[np.ix_(rows[kept], block["columns"])]
        unit_coef = np.concatenate([-omega, np.full(n_treated, 1.0 / n_treated)])
        time_coef = np.concatenate([-lam, np.full(n_post, 1.0 / n_post)])
        taus.append(float(unit_coef @ Y @ time_coef))
        weights.append(n_treated * n_post)

    if not taus:
        return float("nan")
    w = np.asarray(weights, dtype=float)
    return float(np.asarray(taus) @ (w / w.sum()))


def _staggered_se(
    estimate: StaggeredEstimate,
    method: str = "bootstrap",
    replications: int = 200,
    random_state: RandomState = None,
) -> float:
    panel = estimate.panel
    if panel is None:
        raise ValueError("this estimate does not carry its panel; cannot resample")
    rng = np.random.default_rng(random_state)
    blocks = _cohort_blocks(panel, estimate.control_pool)
    N = panel.Y.shape[0]
    treated = panel.adoption >= 0

    if method == "jackknife":
        if int((~treated).sum()) <= 1 or int(treated.sum()) <= 1:
            return float("nan")
        u = np.empty(N)
        for i in range(N):
            keep = np.ones(N, dtype=bool)
            keep[i] = False
            u[i] = _att_with_fixed_weights(panel, blocks, estimate.estimates, keep)
        u = u[np.isfinite(u)]
        n = u.size
        if n < 2:
            return float("nan")
        return float(np.sqrt(((n - 1) / n) * (n - 1) * np.var(u, ddof=1)))

    if method == "bootstrap":
        draws = np.empty(replications)
        count = guard = 0
        while count < replications:
            guard += 1
            if guard > max(1000, 100 * replications):
                raise RuntimeError("bootstrap failed to draw usable samples")
            ind = np.sort(rng.integers(0, N, size=N))
            keep_counts = np.bincount(ind, minlength=N)
            resampled = _bootstrap_att(panel, estimate, keep_counts)
            if not np.isfinite(resampled):
                continue
            draws[count] = resampled
            count += 1
        return float(np.sqrt((replications - 1) / replications) * np.std(draws, ddof=1))

    if method == "placebo":
        n_treated = int(treated.sum())
        control_rows = np.flatnonzero(~treated)
        if control_rows.size <= n_treated:
            raise ValueError(
                "must have more never-treated units than treated units to use "
                "the placebo se"
            )
        adoption_pattern = panel.adoption[treated]
        draws = np.empty(replications)
        for b in range(replications):
            shuffled = rng.permutation(control_rows)
            placebo_rows = shuffled[:n_treated]
            fake = np.full(N, -1, dtype=int)
            fake[placebo_rows] = adoption_pattern
            sub = np.concatenate([shuffled[n_treated:], placebo_rows])
            fake_panel = StaggeredPanel(
                Y=panel.Y[sub],
                adoption=fake[sub],
                units=panel.units[sub],
                time=panel.time,
                X=None if panel.X is None else panel.X[sub],
            )
            draws[b] = staggered_synthdid_estimate(
                fake_panel, estimator=estimate.estimator, control_pool=estimate.control_pool
            ).att
        return float(np.sqrt((replications - 1) / replications) * np.std(draws, ddof=1))

    raise ValueError(f"unknown method {method!r}")


def _bootstrap_att(
    panel: StaggeredPanel, estimate: StaggeredEstimate, counts: np.ndarray
) -> float:
    """Aggregate ATT on a bootstrap resample of units (weights re-estimated)."""
    ind = np.repeat(np.arange(len(counts)), counts)
    adoption = panel.adoption[ind]
    if not np.any(adoption >= 0) or not np.any(adoption < 0):
        return float("nan")
    ind = ind[np.argsort(adoption, kind="stable")]  # never-treated first, then cohorts
    resampled = StaggeredPanel(
        Y=panel.Y[ind],
        adoption=panel.adoption[ind],
        units=np.arange(len(ind)),
        time=panel.time,
        X=None if panel.X is None else panel.X[ind],
    )
    try:
        return staggered_synthdid_estimate(
            resampled, estimator=estimate.estimator, control_pool=estimate.control_pool
        ).att
    except ValueError:
        return float("nan")
