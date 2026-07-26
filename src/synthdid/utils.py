"""Panel-data helpers: long-to-wide reshaping and small numerical utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = [
    "PanelData",
    "panel_matrices",
    "collapsed_form",
    "pairwise_sum_decreasing",
    "sum_normalize",
    "random_low_rank",
]

ColumnRef = Union[int, str]


@dataclass
class PanelData:
    """A balanced panel reshaped into the block structure synthdid expects.

    Rows of ``Y`` are units with the ``N0`` controls first; columns are time
    periods with the ``T0`` pre-treatment periods first.

    The object unpacks like the R list it mirrors::

        Y, N0, T0, W = panel_matrices(df)
    """

    Y: np.ndarray
    N0: int
    T0: int
    W: np.ndarray
    units: np.ndarray
    time: np.ndarray
    X: Optional[np.ndarray] = None
    covariate_names: Optional[np.ndarray] = None

    def __iter__(self) -> Iterator[Any]:
        return iter((self.Y, self.N0, self.T0, self.W))

    @property
    def N1(self) -> int:
        """Number of treated units."""
        return self.Y.shape[0] - self.N0

    @property
    def T1(self) -> int:
        """Number of post-treatment periods."""
        return self.Y.shape[1] - self.T0

    @property
    def treated_units(self) -> np.ndarray:
        return self.units[self.N0:]

    def to_frame(self) -> pd.DataFrame:
        """Return ``Y`` as a DataFrame indexed by unit with time as columns."""
        return pd.DataFrame(self.Y, index=self.units, columns=self.time)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PanelData(N={self.Y.shape[0]} units [{self.N0} control, {self.N1} treated], "
            f"T={self.Y.shape[1]} periods [{self.T0} pre, {self.T1} post])"
        )


def _resolve_column(panel: pd.DataFrame, ref: ColumnRef, what: str) -> str:
    """Map a 1-based column position or a column name onto a column name."""
    if isinstance(ref, str):
        if ref not in panel.columns:
            raise ValueError(f"{what} column {ref!r} is not a column of `panel`.")
        return ref
    if isinstance(ref, (int, np.integer)):
        if ref in panel.columns:  # an integer that is literally a column label
            return ref
        if not 1 <= int(ref) <= panel.shape[1]:
            raise ValueError(
                f"{what} column index {ref} is out of range for a panel with "
                f"{panel.shape[1]} columns (indices are 1-based, as in R)."
            )
        return panel.columns[int(ref) - 1]
    raise ValueError(f"{what} column identifier must be an int or a column name.")


def panel_matrices(
    panel: pd.DataFrame,
    unit: ColumnRef = 1,
    time: ColumnRef = 2,
    outcome: ColumnRef = 3,
    treatment: ColumnRef = 4,
    treated_last: bool = True,
    covariates: Optional[Sequence[ColumnRef]] = None,
) -> PanelData:
    """Convert a long balanced panel into the matrices synthdid needs.

    A typical long panel looks like ``[unit, time, outcome, treatment]``.  The
    estimator requires a *balanced* panel with *simultaneous* adoption: every
    unit is observed in every period, and every treated unit starts treatment in
    the same period.

    Parameters
    ----------
    panel : DataFrame
        The long panel.
    unit, time, outcome, treatment : int or str
        Column positions (1-based, as in R) or column names.
    treated_last : bool
        Sort rows so that treated units come last.  If False, rows are sorted by
        unit identifier only.
    covariates : sequence of int or str, optional
        Extra columns to reshape into an N x T x C array, returned as ``.X`` on
        the result.  This is an extension over the R package, which has no
        covariate support in ``panel.matrices``.

    Returns
    -------
    PanelData
        With fields ``Y``, ``N0``, ``T0``, ``W``, ``units`` and ``time``.

    Raises
    ------
    ValueError
        If the panel is unbalanced, has missing values, has no variation in
        treatment, or if treatment adoption is not simultaneous.
    """
    if not isinstance(panel, pd.DataFrame):
        raise ValueError("Unsupported input type `panel`: expected a pandas DataFrame.")

    unit_c = _resolve_column(panel, unit, "unit")
    time_c = _resolve_column(panel, time, "time")
    outcome_c = _resolve_column(panel, outcome, "outcome")
    treatment_c = _resolve_column(panel, treatment, "treatment")
    cov_c = [_resolve_column(panel, c, "covariate") for c in (covariates or [])]

    keep = [unit_c, time_c, outcome_c, treatment_c, *cov_c]
    if len(set(keep)) != len(keep):
        raise ValueError("The same column was given more than one role.")
    data = panel.loc[:, keep].copy()

    if data.isna().to_numpy().any():
        raise ValueError("Missing values in `panel`.")

    treat = data[treatment_c]
    if treat.dtype == bool:
        treat = treat.astype(int)
    treat = pd.to_numeric(treat, errors="coerce")
    if treat.isna().any() or not treat.isin([0, 1]).all():
        raise ValueError("The treatment status should be in 0 or 1.")
    if treat.nunique() == 1:
        raise ValueError("There is no variation in treatment status.")
    data[treatment_c] = treat.astype(int)

    counts = data.groupby([unit_c, time_c], observed=True, sort=False).size()
    n_units = data[unit_c].nunique()
    n_times = data[time_c].nunique()
    if len(counts) != n_units * n_times or not (counts == 1).all():
        raise ValueError(
            "Input `panel` must be a balanced panel: it must have an observation "
            "for every unit at every time."
        )

    Y_df = data.pivot(index=unit_c, columns=time_c, values=outcome_c).sort_index(
        axis=0
    ).sort_index(axis=1)
    W_df = data.pivot(index=unit_c, columns=time_c, values=treatment_c).sort_index(
        axis=0
    ).sort_index(axis=1)

    Y = Y_df.to_numpy(dtype=float)
    W = W_df.to_numpy(dtype=int)
    units = Y_df.index.to_numpy()
    times = Y_df.columns.to_numpy()

    ever_treated = W.any(axis=1)
    treated_periods = np.flatnonzero(W.any(axis=0))
    T0 = int(treated_periods[0])  # number of periods before anybody is treated
    N0 = int((~ever_treated).sum())

    ok = (
        np.all(W[~ever_treated] == 0)
        and np.all(W[:, :T0] == 0)
        and np.all(W[ever_treated, T0:] == 1)
    )
    if not ok:
        raise ValueError(
            "The package cannot use this data. Treatment adoption is not simultaneous."
        )

    if treated_last:
        order = np.lexsort((np.arange(len(units)), W[:, T0]))
    else:
        order = np.arange(len(units))

    result = PanelData(
        Y=np.ascontiguousarray(Y[order]),
        N0=N0,
        T0=T0,
        W=np.ascontiguousarray(W[order]),
        units=units[order],
        time=times,
    )

    if cov_c:
        X = np.stack(
            [
                data.pivot(index=unit_c, columns=time_c, values=c)
                .sort_index(axis=0)
                .sort_index(axis=1)
                .to_numpy(dtype=float)[order]
                for c in cov_c
            ],
            axis=2,
        )
        result.X = X
        result.covariate_names = np.asarray(cov_c)
    return result


def collapsed_form(Y: np.ndarray, N0: int, T0: int) -> np.ndarray:
    """Collapse ``Y`` to (N0+1) x (T0+1) by averaging treated rows and post columns."""
    Y = np.asarray(Y, dtype=float)
    N, T = Y.shape
    top = np.column_stack([Y[:N0, :T0], Y[:N0, T0:].mean(axis=1)])
    bottom = np.append(Y[N0:, :T0].mean(axis=0), Y[N0:, T0:].mean())
    return np.vstack([top, bottom])


def pairwise_sum_decreasing(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sum two decreasing objective traces, extending the shorter with its minimum.

    The objective traces returned by the solvers stop at different iterations.
    Where one has stopped we treat its value as constant at its final (smallest)
    value, matching ``pairwise.sum.decreasing`` in the R package.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        return y.copy()
    if y.size == 0:
        return x.copy()
    n = max(x.size, y.size)
    xf = np.full(n, np.nanmin(x))
    yf = np.full(n, np.nanmin(y))
    xf[: x.size] = x
    yf[: y.size] = y
    return xf + yf


def sum_normalize(x: np.ndarray) -> np.ndarray:
    """Normalize to sum one; return uniform weights if the input sums to zero."""
    x = np.asarray(x, dtype=float)
    total = x.sum()
    if total != 0:
        return x / total
    return np.full(x.size, 1.0 / x.size) if x.size else x


def random_low_rank(
    n_0: int = 100,
    n_1: int = 10,
    T_0: int = 120,
    T_1: int = 20,
    tau: float = 1.0,
    sigma: float = 0.5,
    rank: int = 2,
    rho: float = 0.7,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> dict:
    """Generate a low-rank panel with a known treatment effect, for tests and demos.

    Mirrors ``random.low.rank`` in the R package: an ``n x T`` outcome matrix
    built from a rank-``rank`` factor model plus unit and time effects, with
    AR-like correlated Gaussian noise and a constant effect ``tau`` on the
    treated block.

    Returns
    -------
    dict with keys ``Y``, ``L`` (the noiseless signal), ``N0`` and ``T0``.
    """
    rng = np.random.default_rng(random_state)
    n = n_0 + n_1
    T = T_0 + T_1

    var = rho ** np.abs(np.subtract.outer(np.arange(1, T + 1), np.arange(1, T + 1)))
    W = np.outer(np.arange(1, n + 1) > n_0, np.arange(1, T + 1) > T_0).astype(float)
    U = rng.poisson(np.sqrt(rng.permutation(np.arange(1, n + 1)) / n)[:, None]
                    * np.ones((n, rank)))
    V = rng.poisson(np.sqrt(np.arange(1, T + 1) / T)[:, None] * np.ones((T, rank)))
    alpha = np.outer(10 * rng.permutation(np.arange(1, n + 1)) / n, np.ones(T))
    beta = np.outer(np.ones(n), 10 * np.arange(1, T + 1) / T)
    mu = U @ V.T + alpha + beta

    chol = np.linalg.cholesky(var)
    error = rng.standard_normal((n, T)) @ chol.T
    Y = mu + tau * W + sigma * error
    return {"Y": Y, "L": mu, "N0": n_0, "T0": T_0}
