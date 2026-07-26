"""The synthetic difference-in-differences estimator (Arkhangelsky et al. 2021).

Port of ``R/synthdid.R``.  :func:`synthdid_estimate` implements Algorithm 1 of
the paper; :func:`sc_estimate` and :func:`did_estimate` are the same machinery
with the weights restricted so that it reduces to synthetic control and to
plain difference-in-differences respectively.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .solver import (
    contract3,
    sc_weight_fw,
    sc_weight_fw_covariates,
    sparsify_function,
)
from .utils import collapsed_form, pairwise_sum_decreasing

# Matrix products use ``.dot`` rather than ``@``; see the note in solver.py.

__all__ = [
    "Setup",
    "Weights",
    "SynthDIDEstimate",
    "synthdid_estimate",
    "sc_estimate",
    "did_estimate",
    "synthdid_placebo",
    "synthdid_effect_curve",
    "refit",
]


def _as_matrix(Y: Any) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Accept an ndarray or a DataFrame; a DataFrame also supplies row/column labels."""
    if isinstance(Y, pd.DataFrame):
        return (
            Y.to_numpy(dtype=float),
            Y.index.to_numpy(),
            Y.columns.to_numpy(),
        )
    arr = np.asarray(Y, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Y must be a 2-d matrix, got shape {arr.shape}")
    return arr, None, None


@dataclass
class Setup:
    """The estimation problem: outcomes, covariates and the treated block."""

    Y: np.ndarray
    X: np.ndarray
    N0: int
    T0: int
    unit_names: Optional[np.ndarray] = None
    time_labels: Optional[np.ndarray] = None

    @property
    def N1(self) -> int:
        return self.Y.shape[0] - self.N0

    @property
    def T1(self) -> int:
        return self.Y.shape[1] - self.T0

    def unit_labels(self) -> np.ndarray:
        if self.unit_names is not None:
            return np.asarray(self.unit_names)
        return np.arange(1, self.Y.shape[0] + 1)

    def period_labels(self) -> np.ndarray:
        if self.time_labels is not None:
            return np.asarray(self.time_labels)
        return np.arange(1, self.Y.shape[1] + 1)


@dataclass
class Weights:
    """Estimated weights: ``omega`` over control units, ``lambda_`` over pre-periods."""

    lambda_: Optional[np.ndarray] = None
    omega: Optional[np.ndarray] = None
    beta: Optional[np.ndarray] = None
    vals: Optional[np.ndarray] = None
    lambda_vals: Optional[np.ndarray] = None
    omega_vals: Optional[np.ndarray] = None

    #: Aliases so ``weights["lambda"]`` works like the R list it mirrors.
    _ALIASES = {"lambda": "lambda_", "lambda.vals": "lambda_vals", "omega.vals": "omega_vals"}

    @property
    def time(self) -> Optional[np.ndarray]:
        """Alias for ``lambda_``: the weight on each pre-treatment period."""
        return self.lambda_

    @property
    def unit(self) -> Optional[np.ndarray]:
        """Alias for ``omega``: the weight on each control unit."""
        return self.omega

    def __getitem__(self, key: str) -> Any:
        return getattr(self, self._ALIASES.get(key, key))

    def copy(self) -> "Weights":
        return replace(self)


def _coerce_weights(weights: Union[None, Weights, Mapping[str, Any]]) -> Weights:
    if weights is None:
        return Weights()
    if isinstance(weights, Weights):
        return weights.copy()
    if isinstance(weights, Mapping):
        out = Weights()
        for key, value in weights.items():
            name = Weights._ALIASES.get(key, key)
            if not hasattr(out, name):
                raise ValueError(f"Unknown weight field {key!r}")
            setattr(out, name, None if value is None else np.asarray(value, dtype=float))
        return out
    raise TypeError("`weights` must be a Weights instance, a mapping, or None")


class SynthDIDEstimate(float):
    """A treatment-effect estimate that also carries the fitted weights.

    Behaves as a plain ``float`` (so it can be printed, compared and used in
    arithmetic) while exposing the pieces needed for inference and plotting:
    :attr:`weights`, :attr:`setup` and :attr:`opts`.
    """

    __slots__ = ("estimator", "weights", "setup", "opts")

    def __new__(
        cls,
        value: float,
        estimator: str = "synthdid_estimate",
        weights: Optional[Weights] = None,
        setup: Optional[Setup] = None,
        opts: Optional[dict] = None,
    ) -> "SynthDIDEstimate":
        self = super().__new__(cls, float(value))
        self.estimator = estimator
        self.weights = weights if weights is not None else Weights()
        self.setup = setup
        self.opts = dict(opts or {})
        return self

    # -- inference ---------------------------------------------------------
    def vcov(self, method: str = "bootstrap", replications: int = 200, **kwargs) -> np.ndarray:
        """Variance of the estimate; see :func:`synthdid.vcov.vcov`."""
        from .vcov import vcov as _vcov

        return _vcov(self, method=method, replications=replications, **kwargs)

    def se(self, method: str = "bootstrap", replications: int = 200, **kwargs) -> float:
        """Standard error of the estimate."""
        return float(np.sqrt(self.vcov(method=method, replications=replications, **kwargs).item()))

    def ci(
        self,
        level: float = 0.95,
        method: str = "bootstrap",
        replications: int = 200,
        **kwargs,
    ) -> tuple[float, float]:
        """Normal-approximation confidence interval for the treatment effect."""
        from scipy.stats import norm

        se = self.se(method=method, replications=replications, **kwargs)
        z = norm.ppf(0.5 + level / 2.0)
        return (float(self) - z * se, float(self) + z * se)

    # -- diagnostics -------------------------------------------------------
    def summary(self, weight_digits: int = 3, fast: bool = False, **kwargs):
        """Estimate, standard error, dimensions and the leading weights."""
        from .summary import summary as _summary

        return _summary(self, weight_digits=weight_digits, fast=fast, **kwargs)

    def controls(self, mass: float = 0.9, weight_type: str = "omega") -> pd.Series:
        from .summary import synthdid_controls

        return synthdid_controls(self, mass=mass, weight_type=weight_type)

    def placebo(self, treated_fraction: Optional[float] = None) -> "SynthDIDEstimate":
        return synthdid_placebo(self, treated_fraction=treated_fraction)

    def effect_curve(self) -> np.ndarray:
        return synthdid_effect_curve(self)

    # -- plotting ----------------------------------------------------------
    def plot(self, **kwargs):
        """Trajectory plot with the 2x2 diff-in-diff diagram overlaid."""
        from .plot import synthdid_plot

        return synthdid_plot(self, **kwargs)

    def units_plot(self, **kwargs):
        from .plot import synthdid_units_plot

        return synthdid_units_plot(self, **kwargs)

    def rmse_plot(self, **kwargs):
        from .plot import synthdid_rmse_plot

        return synthdid_rmse_plot(self, **kwargs)

    def placebo_plot(self, **kwargs):
        from .plot import synthdid_placebo_plot

        return synthdid_placebo_plot(self, **kwargs)

    # -- serialization -----------------------------------------------------
    def __reduce__(self):
        # float subclasses with __slots__ need this to survive pickle/deepcopy
        return (
            _rebuild_estimate,
            (float(self), self.estimator, self.weights, self.setup, self.opts),
        )

    # -- representation ----------------------------------------------------
    def format(self, se_method: str = "jackknife") -> str:
        from .summary import format_estimate

        return format_estimate(self, se_method=se_method)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        # Deliberately does not compute a standard error: unlike R's print
        # method, repr here stays cheap and side-effect free. Use .format() for
        # the R-style one-liner, or .summary() for the full report.
        if self.setup is None:
            return f"{self.estimator}: {float(self):.6f}"
        s = self.setup
        omega = np.asarray(self.weights.omega, dtype=float)
        lam = np.asarray(self.weights.lambda_, dtype=float)
        n0_eff = 1.0 / np.sum(omega ** 2) if np.any(omega) else float("inf")
        t0_eff = 1.0 / np.sum(lam ** 2) if np.any(lam) else float("inf")
        return (
            f"{self.estimator}: {float(self):.4f}. "
            f"Effective N0/N0 = {n0_eff:.1f}/{s.N0}. "
            f"Effective T0/T0 = {t0_eff:.1f}/{s.T0}. "
            f"N1,T1 = {s.N1},{s.T1}."
        )

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.__repr__()


def _rebuild_estimate(value, estimator, weights, setup, opts) -> SynthDIDEstimate:
    """Unpickling hook for :class:`SynthDIDEstimate`."""
    return SynthDIDEstimate(value, estimator=estimator, weights=weights, setup=setup, opts=opts)


def synthdid_estimate(
    Y: Any,
    N0: int,
    T0: int,
    X: Optional[np.ndarray] = None,
    noise_level: Optional[float] = None,
    eta_omega: Optional[float] = None,
    eta_lambda: float = 1e-6,
    zeta_omega: Optional[float] = None,
    zeta_lambda: Optional[float] = None,
    omega_intercept: bool = True,
    lambda_intercept: bool = True,
    weights: Union[None, Weights, Mapping[str, Any]] = None,
    update_omega: Optional[bool] = None,
    update_lambda: Optional[bool] = None,
    min_decrease: Optional[float] = None,
    max_iter: int = 10_000,
    sparsify: Optional[Callable[[np.ndarray], np.ndarray]] = sparsify_function,
    max_iter_pre_sparsify: int = 100,
    unit_names: Optional[Sequence] = None,
    time_labels: Optional[Sequence] = None,
) -> SynthDIDEstimate:
    """Synthetic difference-in-differences estimate of the ATT on a treated block.

    Implements Algorithm 1 of Arkhangelsky, Athey, Hirshberg, Imbens and Wager,
    *Synthetic Difference-in-Differences* (American Economic Review, 2021).

    Parameters
    ----------
    Y : array of shape (N, T) or DataFrame
        The outcome matrix.  Rows ``0..N0-1`` are the control units and columns
        ``0..T0-1`` the pre-treatment periods.  A DataFrame's index and columns
        are kept as unit and period labels.
    N0 : int
        Number of control units.
    T0 : int
        Number of pre-treatment periods.
    X : array of shape (N, T, C), optional
        Time-varying covariates.
    noise_level : float, optional
        Estimate of the noise standard deviation.  Defaults to the standard
        deviation of first differences of the control/pre-treatment block.
    eta_omega : float, optional
        Sets ``zeta_omega = eta_omega * noise_level``.  Defaults to
        ``(N1 * T1) ** 0.25``.
    eta_lambda : float
        Analogous for lambda; defaults to an infinitesimal ``1e-6``.
    zeta_omega, zeta_lambda : float, optional
        Override the regularization levels directly.
    omega_intercept, lambda_intercept : bool
        Fit an intercept when solving for the respective weights.
    weights : Weights or mapping, optional
        Supply ``lambda_``/``omega`` to hold weights fixed (or to initialize
        them, together with ``update_lambda``/``update_omega``).
    update_omega, update_lambda : bool, optional
        Whether to solve for the weights.  Default: solve unless the weight was
        passed in.
    min_decrease : float, optional
        Stop when an iteration decreases penalized MSE by less than
        ``min_decrease ** 2``.  Defaults to ``1e-5 * noise_level``.
    max_iter : int
        Iteration cap for the solver.
    sparsify : callable or None
        Applied to the first-round solution before a second round of
        optimization, to encourage sparse weights.  Pass ``None`` to skip.
    max_iter_pre_sparsify : int
        Iteration cap for the first round.
    unit_names, time_labels : sequence, optional
        Labels used by summaries and plots when ``Y`` is a plain array.

    Returns
    -------
    SynthDIDEstimate
        A float carrying ``.weights``, ``.setup`` and ``.opts``.

    Examples
    --------
    >>> from synthdid import load_california_prop99, panel_matrices, synthdid_estimate
    >>> panel = panel_matrices(load_california_prop99())
    >>> tau = synthdid_estimate(panel.Y, panel.N0, panel.T0)
    >>> round(float(tau), 3)
    -15.604
    """
    Y, y_units, y_times = _as_matrix(Y)
    if unit_names is not None:
        y_units = np.asarray(unit_names)
    if time_labels is not None:
        y_times = np.asarray(time_labels)

    N, T = Y.shape
    N0, T0 = int(N0), int(T0)
    if not N > N0:
        raise ValueError(f"Need at least one treated unit: N={N} but N0={N0}")
    if not T > T0:
        raise ValueError(f"Need at least one post-treatment period: T={T} but T0={T0}")
    N1, T1 = N - N0, T - T0

    if X is None:
        X = np.zeros((N, T, 0))
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[:, :, None]
    if X.ndim != 3 or X.shape[:2] != (N, T):
        raise ValueError(f"X must have shape ({N}, {T}, C); got {X.shape}")

    w = _coerce_weights(weights)
    if w.lambda_ is not None and w.lambda_.size != T0:
        raise ValueError(f"weights.lambda_ must have length T0={T0}")
    if w.omega is not None and w.omega.size != N0:
        raise ValueError(f"weights.omega must have length N0={N0}")
    if update_lambda is None:
        update_lambda = w.lambda_ is None
    if update_omega is None:
        update_omega = w.omega is None
    if w.lambda_ is None and not update_lambda:
        raise ValueError("update_lambda=False requires passing weights.lambda_")
    if w.omega is None and not update_omega:
        raise ValueError("update_omega=False requires passing weights.omega")

    if noise_level is None:
        noise_level = float(np.std(np.diff(Y[:N0, :T0], axis=1), ddof=1))
    if eta_omega is None:
        eta_omega = (N1 * T1) ** 0.25
    if zeta_omega is None:
        zeta_omega = eta_omega * noise_level
    if zeta_lambda is None:
        zeta_lambda = eta_lambda * noise_level
    if min_decrease is None:
        min_decrease = 1e-5 * noise_level
    if sparsify is None:
        max_iter_pre_sparsify = max_iter

    if X.shape[2] == 0:
        if w.beta is not None and w.beta.size > 0:
            raise ValueError(
                "weights.beta was given but no covariates X were passed; "
                "coefficients have nothing to multiply."
            )
        w.beta = None
        w.vals = None
        w.lambda_vals = None
        w.omega_vals = None
        if update_lambda:
            Yc = collapsed_form(Y, N0, T0)
            opt = sc_weight_fw(
                Yc[:N0, :],
                zeta=zeta_lambda,
                intercept=lambda_intercept,
                lambda_=w.lambda_,
                min_decrease=min_decrease,
                max_iter=max_iter_pre_sparsify,
            )
            if sparsify is not None:
                opt = sc_weight_fw(
                    Yc[:N0, :],
                    zeta=zeta_lambda,
                    intercept=lambda_intercept,
                    lambda_=sparsify(opt["lambda"]),
                    min_decrease=min_decrease,
                    max_iter=max_iter,
                )
            w.lambda_ = opt["lambda"]
            w.lambda_vals = opt["vals"]
            w.vals = opt["vals"]
        if update_omega:
            Yc = collapsed_form(Y, N0, T0)
            opt = sc_weight_fw(
                Yc[:, :T0].T,
                zeta=zeta_omega,
                intercept=omega_intercept,
                lambda_=w.omega,
                min_decrease=min_decrease,
                max_iter=max_iter_pre_sparsify,
            )
            if sparsify is not None:
                opt = sc_weight_fw(
                    Yc[:, :T0].T,
                    zeta=zeta_omega,
                    intercept=omega_intercept,
                    lambda_=sparsify(opt["lambda"]),
                    min_decrease=min_decrease,
                    max_iter=max_iter,
                )
            w.omega = opt["lambda"]
            w.omega_vals = opt["vals"]
            w.vals = (
                opt["vals"]
                if w.vals is None
                else pairwise_sum_decreasing(w.vals, opt["vals"])
            )
    else:
        Yc = collapsed_form(Y, N0, T0)
        Xc = np.stack([collapsed_form(X[:, :, c], N0, T0) for c in range(X.shape[2])], axis=2)
        solved = sc_weight_fw_covariates(
            Yc,
            Xc,
            zeta_lambda=zeta_lambda,
            zeta_omega=zeta_omega,
            lambda_intercept=lambda_intercept,
            omega_intercept=omega_intercept,
            min_decrease=min_decrease,
            max_iter=max_iter,
            lambda_=w.lambda_,
            omega=w.omega,
            update_lambda=update_lambda,
            update_omega=update_omega,
        )
        w = Weights(
            lambda_=solved["lambda"],
            omega=solved["omega"],
            beta=solved["beta"],
            vals=solved["vals"],
        )

    X_beta = contract3(X, w.beta)
    unit_coef = np.concatenate([-w.omega, np.full(N1, 1.0 / N1)])
    time_coef = np.concatenate([-w.lambda_, np.full(T1, 1.0 / T1)])
    estimate = float(unit_coef.dot(Y - X_beta).dot(time_coef))

    return SynthDIDEstimate(
        estimate,
        estimator="synthdid_estimate",
        weights=w,
        setup=Setup(Y=Y, X=X, N0=N0, T0=T0, unit_names=y_units, time_labels=y_times),
        opts={
            "zeta_omega": zeta_omega,
            "zeta_lambda": zeta_lambda,
            "omega_intercept": omega_intercept,
            "lambda_intercept": lambda_intercept,
            "update_omega": update_omega,
            "update_lambda": update_lambda,
            "min_decrease": min_decrease,
            "max_iter": max_iter,
        },
    )


def sc_estimate(Y: Any, N0: int, T0: int, eta_omega: float = 1e-6, **kwargs) -> SynthDIDEstimate:
    """Synthetic control estimate: the same solver with ``lambda`` pinned to zero.

    By default this uses only infinitesimal ridge regularization and no unit
    intercept, reproducing Abadie-Diamond-Hainmueller synthetic control.
    """
    T0 = int(T0)
    kwargs.setdefault("weights", {"lambda": np.zeros(T0)})
    kwargs.setdefault("omega_intercept", False)
    estimate = synthdid_estimate(Y, N0, T0, eta_omega=eta_omega, **kwargs)
    estimate.estimator = "sc_estimate"
    return estimate


def did_estimate(Y: Any, N0: int, T0: int, **kwargs) -> SynthDIDEstimate:
    """Difference-in-differences estimate: uniform unit and time weights."""
    N0, T0 = int(N0), int(T0)
    kwargs.setdefault(
        "weights", {"lambda": np.full(T0, 1.0 / T0), "omega": np.full(N0, 1.0 / N0)}
    )
    estimate = synthdid_estimate(Y, N0, T0, **kwargs)
    estimate.estimator = "did_estimate"
    return estimate


_ESTIMATORS: dict[str, Callable[..., SynthDIDEstimate]] = {
    "synthdid_estimate": synthdid_estimate,
    "sc_estimate": sc_estimate,
    "did_estimate": did_estimate,
}


def refit(estimate: SynthDIDEstimate, **overrides) -> SynthDIDEstimate:
    """Re-run :func:`synthdid_estimate` with this estimate's tuning options.

    Resampling-based variance estimators call this on perturbed data.  As in the
    R package, the generic estimator is always used: the estimator-specific
    behaviour of :func:`sc_estimate` and :func:`did_estimate` is fully captured
    by the recorded options plus the weights that are passed back in.
    """
    kwargs = dict(estimate.opts)
    kwargs.update(overrides)
    out = synthdid_estimate(**kwargs)
    out.estimator = estimate.estimator
    return out


def synthdid_placebo(
    estimate: SynthDIDEstimate, treated_fraction: Optional[float] = None
) -> SynthDIDEstimate:
    """Refit the estimator on pre-treatment data only, with a placebo treatment date.

    Parameters
    ----------
    estimate : SynthDIDEstimate
    treated_fraction : float, optional
        Fraction of the pre-treatment periods to treat as the placebo
        post-period.  Defaults to the actual post/total ratio.
    """
    setup = estimate.setup
    if treated_fraction is None:
        treated_fraction = 1.0 - setup.T0 / setup.Y.shape[1]
    placebo_T0 = int(np.floor(setup.T0 * (1.0 - treated_fraction)))

    kwargs = dict(estimate.opts)
    if estimate.estimator != "synthdid_estimate":
        # sc_estimate / did_estimate rebuild their own fixed weights for the new T0
        kwargs.pop("update_lambda", None)
        kwargs.pop("update_omega", None)
    out = _ESTIMATORS[estimate.estimator](
        Y=setup.Y[:, : setup.T0],
        N0=setup.N0,
        T0=placebo_T0,
        X=setup.X[:, : setup.T0, :],
        unit_names=setup.unit_names,
        time_labels=None if setup.time_labels is None else setup.time_labels[: setup.T0],
        **kwargs,
    )
    out.estimator = estimate.estimator
    return out


def synthdid_effect_curve(estimate: SynthDIDEstimate) -> np.ndarray:
    """The per-period treatment effects that average to the estimate.

    Returns an array of length ``T1``: for each post-treatment period, the
    difference between treated and synthetic-control outcomes, net of the
    lambda-weighted pre-treatment difference.
    """
    setup = estimate.setup
    w = estimate.weights
    Y = setup.Y - contract3(setup.X, w.beta)
    N1 = setup.N1
    T1 = setup.T1

    unit_coef = np.concatenate([-w.omega, np.full(N1, 1.0 / N1)])
    tau_sc = unit_coef.dot(Y)
    return tau_sc[setup.T0:] - tau_sc[: setup.T0] @ w.lambda_
