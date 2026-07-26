"""Matplotlib versions of the diagnostic plots in the R package.

:func:`synthdid_plot` reproduces the signature synthdid figure: treated and
synthetic-control trajectories, the 2x2 difference-in-differences parallelogram
overlaid, the treatment effect drawn as an arrow, and the time weights
``lambda`` shown as a ribbon along the bottom.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from .estimate import SynthDIDEstimate, synthdid_placebo
from .solver import contract3

__all__ = [
    "synthdid_plot",
    "synthdid_units_plot",
    "synthdid_rmse_plot",
    "synthdid_placebo_plot",
]

Estimates = Union[SynthDIDEstimate, Sequence[SynthDIDEstimate], Mapping[str, SynthDIDEstimate]]

#: ggplot2's default two-colour hue palette, so figures match the R package.
CONTROL_COLOR = "#F8766D"
TREATED_COLOR = "#00BFC4"
_ESTIMATE_COLORS = ("#F8766D", "#7CAE00", "#00BFC4", "#C77CFF", "#FFB000", "#00A9FF")


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Plotting requires matplotlib. Install it with "
            "`pip install synthdid-py[plot]`."
        ) from exc
    return plt


def _as_named(estimates: Estimates) -> dict[str, SynthDIDEstimate]:
    if isinstance(estimates, SynthDIDEstimate):
        return {"estimate 1": estimates}
    if isinstance(estimates, Mapping):
        return dict(estimates)
    return {f"estimate {i + 1}": est for i, est in enumerate(estimates)}


def _numeric_time(estimate: SynthDIDEstimate) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Return x coordinates for the time axis, plus tick labels when not numeric."""
    labels = estimate.setup.period_labels()
    try:
        x = np.asarray(labels, dtype=float)
        if np.all(np.isfinite(x)):
            return x, None
    except (TypeError, ValueError):
        pass
    try:
        import pandas as pd

        as_dates = pd.to_datetime(labels)
        return np.asarray(as_dates.astype("int64") / 86_400_000_000_000, dtype=float), np.asarray(
            [str(d.date()) for d in as_dates]
        )
    except Exception:
        return np.arange(1.0, len(labels) + 1.0), np.asarray([str(v) for v in labels])


class _PlotGeometry:
    """The coordinates of every element of the synthdid diagram, for one estimate."""

    def __init__(self, estimate: SynthDIDEstimate, overlay: float, se: float):
        setup = estimate.setup
        w = estimate.weights
        Y = setup.Y - contract3(setup.X, w.beta)
        N0, T0 = setup.N0, setup.T0
        N1, T1 = setup.N1, setup.T1

        self.lambda_synth = np.concatenate([w.lambda_, np.zeros(T1)])
        self.lambda_target = np.concatenate([np.zeros(T0), np.full(T1, 1.0 / T1)])
        omega_synth = np.concatenate([w.omega, np.zeros(N1)])
        omega_target = np.concatenate([np.zeros(N0), np.full(N1, 1.0 / N1)])

        self.is_sc = bool(np.all(w.lambda_ == 0)) or overlay == 1
        offset = overlay * float((omega_target - omega_synth) @ Y @ self.lambda_synth)

        self.obs_trajectory = omega_target @ Y
        self.syn_trajectory = omega_synth @ Y + offset
        self.Y = Y
        self.N0, self.T0, self.N1, self.T1 = N0, T0, N1, T1
        self.se = se

        self.treated_post = float(omega_target @ Y @ self.lambda_target)
        self.treated_pre = float(omega_target @ Y @ self.lambda_synth)
        self.control_post = float(omega_synth @ Y @ self.lambda_target) + offset
        self.control_pre = float(omega_synth @ Y @ self.lambda_synth) + offset
        self.sdid_post = self.control_post + self.treated_pre - self.control_pre

        self.time, self.tick_labels = _numeric_time(estimate)
        self.pre_time = float(self.lambda_synth @ self.time)
        self.post_time = float(self.lambda_target @ self.time)
        self.onset = float(self.time[T0 - 1])

    def ribbon(self, scale: float, comparable: bool) -> tuple[np.ndarray, float, np.ndarray]:
        """Baseline and heights for the lambda ribbon drawn under the trajectories."""
        if comparable:
            span = self.obs_trajectory.max() - self.obs_trajectory.min()
            height = span / scale
            bottom = self.obs_trajectory.min() - height
            top = bottom + height * self.lambda_synth[: self.T0]
        else:
            both = np.concatenate([self.obs_trajectory, self.syn_trajectory])
            height = (both.max() - both.min()) / scale
            bottom = both.min() - height
            peak = self.lambda_synth.max()
            scaled = self.lambda_synth[: self.T0] / peak if peak > 0 else self.lambda_synth[: self.T0]
            top = bottom + height * scaled
        return self.time[: self.T0], bottom, top


def synthdid_plot(
    estimates: Estimates,
    treated_name: str = "treated",
    control_name: str = "synthetic control",
    spaghetti_units: Sequence = (),
    facet: Optional[Sequence] = None,
    facet_vertical: bool = True,
    lambda_comparable: Optional[bool] = None,
    overlay: float = 0.0,
    lambda_plot_scale: float = 3.0,
    trajectory_linestyle: str = "-",
    effect_curvature: float = 0.3,
    line_width: float = 1.2,
    guide_linestyle: str = "--",
    point_size: float = 5.0,
    trajectory_alpha: float = 0.7,
    diagram_alpha: float = 0.95,
    effect_alpha: float = 0.95,
    onset_alpha: float = 0.3,
    ci_alpha: float = 0.3,
    spaghetti_line_width: float = 0.5,
    spaghetti_label_size: float = 7.0,
    spaghetti_line_alpha: float = 0.3,
    spaghetti_label_alpha: float = 0.5,
    se_method: Optional[str] = "jackknife",
    se_replications: int = 200,
    alpha_multiplier: Optional[Sequence[float]] = None,
    ax: Optional[Any] = None,
    figsize: Optional[tuple[float, float]] = None,
):
    """Plot trajectories with the 2x2 diff-in-diff diagram of the estimator overlaid.

    The treatment effect is the vertical arrow at the (lambda-weighted)
    post-treatment time; the dashed parallelogram shows the counterfactual that
    synthdid extrapolates from the control trajectory.  Time weights are drawn
    as a ribbon along the bottom.  For synthetic-control estimates (all lambda
    zero) only the trajectories and effect arrow are drawn.

    Parameters
    ----------
    estimates : estimate, sequence or mapping of estimates
        Several estimates are drawn in separate panels unless ``facet`` groups
        them together.
    treated_name, control_name : str
        Legend labels.
    spaghetti_units : sequence
        Unit labels (elements of ``setup.unit_names``) to draw individually.
    facet : sequence, optional
        One entry per estimate naming the panel it belongs in.  Estimates
        sharing an entry are overlaid.  Default: one panel per estimate.
    facet_vertical : bool
        Stack panels vertically rather than side by side.
    lambda_comparable : bool, optional
        Scale the lambda ribbons identically across panels.  Defaults to True
        when ``facet`` is given.
    overlay : float in [0, 1]
        Shift the control trajectory toward the treated one by this fraction of
        the diff-in-diff intercept adjustment; ``1`` overlays them and
        suppresses the diagram.
    se_method : str or None
        Method used for the 95% confidence-interval arrows; ``None`` omits them.
    ax : Axes or array of Axes, optional
        Draw into existing axes instead of creating a figure.
    figsize : tuple, optional

    Returns
    -------
    (Figure, Axes or ndarray of Axes)
    """
    plt = _require_matplotlib()
    named = _as_named(estimates)
    names = list(named)
    n_est = len(names)

    if alpha_multiplier is None:
        alpha_multiplier = np.ones(n_est)
    alpha_multiplier = np.asarray(alpha_multiplier, dtype=float)
    if lambda_comparable is None:
        lambda_comparable = facet is not None

    facet_keys = list(facet) if facet is not None else names
    panels: dict[Any, list[int]] = {}
    for i, key in enumerate(facet_keys):
        panels.setdefault(key, []).append(i)
    panel_keys = list(panels)

    if ax is None:
        n = len(panel_keys)
        if figsize is None:
            figsize = (8.0, 4.0 * n) if facet_vertical else (6.0 * n, 4.5)
        shape = (n, 1) if facet_vertical else (1, n)
        # stacked panels get their own y scale, side-by-side panels share one,
        # matching facet_grid(scales='free_y') vs facet_grid(. ~ facet) in R
        fig, axes = plt.subplots(
            *shape, figsize=figsize, squeeze=False, sharey=not facet_vertical
        )
        axes = axes.ravel()
    else:
        axes = np.atleast_1d(np.asarray(ax, dtype=object)).ravel()
        fig = axes[0].figure
        if len(axes) < len(panel_keys):
            raise ValueError(f"need {len(panel_keys)} axes, got {len(axes)}")

    ses = {}
    for name, est in named.items():
        if se_method in (None, "none"):
            ses[name] = np.nan
        else:
            kwargs = {} if se_method == "jackknife" else {"replications": se_replications}
            ses[name] = est.se(method=se_method, **kwargs)

    for panel_index, key in enumerate(panel_keys):
        axis = axes[panel_index]
        members = panels[key]
        one_per_panel = len(members) == 1
        geom = None
        for member in members:
            name = names[member]
            geom = _PlotGeometry(named[name], overlay, ses[name])
            color = (
                CONTROL_COLOR
                if one_per_panel
                else _ESTIMATE_COLORS[member % len(_ESTIMATE_COLORS)]
            )
            _draw_estimate(
                axis,
                geom,
                control_color=color,
                treated_color=TREATED_COLOR,
                control_label=control_name if one_per_panel else name,
                treated_label=treated_name,
                show_treated=member == members[0],
                show=float(alpha_multiplier[member]),
                lambda_comparable=lambda_comparable,
                lambda_plot_scale=lambda_plot_scale,
                trajectory_linestyle=trajectory_linestyle,
                effect_curvature=effect_curvature,
                line_width=line_width,
                guide_linestyle=guide_linestyle,
                point_size=point_size,
                trajectory_alpha=trajectory_alpha,
                diagram_alpha=diagram_alpha,
                effect_alpha=effect_alpha,
                onset_alpha=onset_alpha,
                ci_alpha=ci_alpha,
                spaghetti_units=spaghetti_units,
                spaghetti_line_width=spaghetti_line_width,
                spaghetti_label_size=spaghetti_label_size,
                spaghetti_line_alpha=spaghetti_line_alpha,
                spaghetti_label_alpha=spaghetti_label_alpha,
                unit_names=named[name].setup.unit_labels(),
            )
        if len(panel_keys) > 1 or facet is not None:
            axis.set_title(str(key), fontsize=10)
        axis.legend(loc="upper center", ncol=3, frameon=False, fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
        if geom is None:
            continue
        if geom.tick_labels is not None:
            step = max(1, len(geom.time) // 10)
            axis.set_xticks(geom.time[::step])
            axis.set_xticklabels(geom.tick_labels[::step], rotation=45, ha="right")
        elif np.allclose(geom.time, np.round(geom.time)):
            from matplotlib.ticker import MaxNLocator

            axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    for extra in axes[len(panel_keys):]:
        extra.set_visible(False)
    fig.tight_layout()
    return fig, (axes[0] if len(panel_keys) == 1 else axes[: len(panel_keys)])


def _draw_estimate(
    axis,
    geom: _PlotGeometry,
    control_color: str,
    treated_color: str,
    control_label: str,
    treated_label: str,
    show_treated: bool,
    show: float,
    lambda_comparable: bool,
    lambda_plot_scale: float,
    trajectory_linestyle: str,
    effect_curvature: float,
    line_width: float,
    guide_linestyle: str,
    point_size: float,
    trajectory_alpha: float,
    diagram_alpha: float,
    effect_alpha: float,
    onset_alpha: float,
    ci_alpha: float,
    spaghetti_units: Sequence,
    spaghetti_line_width: float,
    spaghetti_label_size: float,
    spaghetti_line_alpha: float,
    spaghetti_label_alpha: float,
    unit_names: np.ndarray,
) -> None:
    from matplotlib.patches import FancyArrowPatch

    time = geom.time

    if show_treated:
        axis.plot(
            time,
            geom.obs_trajectory,
            color=treated_color,
            linestyle=trajectory_linestyle,
            linewidth=line_width,
            alpha=trajectory_alpha,
            label=treated_label,
            zorder=3,
        )
    axis.plot(
        time,
        geom.syn_trajectory,
        color=control_color,
        linestyle=trajectory_linestyle,
        linewidth=line_width,
        alpha=trajectory_alpha * show,
        label=control_label,
        zorder=3,
    )

    if len(spaghetti_units):
        selected = np.isin(unit_names, np.asarray(spaghetti_units))
        for row, label in zip(geom.Y[selected], np.asarray(unit_names)[selected]):
            axis.plot(
                time,
                row,
                color="black",
                linewidth=spaghetti_line_width,
                alpha=spaghetti_line_alpha * show,
                zorder=1,
            )
            axis.text(
                time[0],
                row[0],
                str(label),
                fontsize=spaghetti_label_size,
                alpha=spaghetti_label_alpha * show,
                ha="right",
                va="center",
            )

    axis.axvline(
        geom.onset, color="black", linewidth=line_width, alpha=onset_alpha * show, zorder=1
    )

    if not geom.is_sc:
        # the lambda ribbon
        x, bottom, top = geom.ribbon(lambda_plot_scale, lambda_comparable)
        axis.fill_between(
            x,
            bottom,
            top,
            step=None,
            facecolor=control_color,
            edgecolor="black",
            linewidth=line_width * 0.5,
            alpha=0.5 * diagram_alpha * show,
            zorder=1,
        )
        # the two arms of the parallelogram
        axis.plot(
            [geom.pre_time, geom.post_time],
            [geom.control_pre, geom.control_post],
            color=control_color,
            linewidth=line_width,
            alpha=diagram_alpha * show,
            zorder=4,
        )
        axis.plot(
            [geom.pre_time, geom.post_time],
            [geom.treated_pre, geom.treated_post],
            color=treated_color,
            linewidth=line_width,
            alpha=diagram_alpha * show,
            zorder=4,
        )
        # the counterfactual arm and the vertical guides
        axis.plot(
            [geom.pre_time, geom.post_time],
            [geom.treated_pre, geom.sdid_post],
            color="black",
            linestyle=guide_linestyle,
            linewidth=line_width,
            alpha=0.6 * diagram_alpha * show,
            zorder=4,
        )
        for x_at, y0, y1 in (
            (geom.pre_time, geom.control_pre, geom.treated_pre),
            (geom.post_time, geom.control_post, geom.sdid_post),
        ):
            axis.plot(
                [x_at, x_at],
                [y0, y1],
                color="black",
                linestyle=guide_linestyle,
                linewidth=line_width,
                alpha=0.5 * diagram_alpha * show,
                zorder=4,
            )
        axis.scatter(
            [geom.pre_time, geom.pre_time, geom.post_time, geom.post_time],
            [geom.treated_pre, geom.control_pre, geom.control_post, geom.treated_post],
            s=point_size ** 2,
            c=[treated_color, control_color, control_color, treated_color],
            alpha=diagram_alpha * show,
            zorder=5,
        )

    axis.scatter(
        [geom.post_time, geom.post_time],
        [geom.treated_post, geom.sdid_post],
        s=point_size ** 2,
        facecolors="none",
        edgecolors=[treated_color, control_color],
        alpha=diagram_alpha * show,
        zorder=5,
    )

    def arrow(y_from: float, alpha: float) -> None:
        if not np.isfinite(y_from) or np.isclose(y_from, geom.treated_post):
            return
        axis.add_patch(
            FancyArrowPatch(
                (geom.post_time, y_from),
                (geom.post_time, geom.treated_post),
                connectionstyle=f"arc3,rad={effect_curvature}",
                arrowstyle="-|>",
                mutation_scale=10,
                color="black",
                linewidth=line_width,
                alpha=alpha * show,
                zorder=6,
            )
        )

    arrow(geom.sdid_post, effect_alpha)
    if np.isfinite(geom.se):
        arrow(geom.sdid_post + 1.96 * geom.se, ci_alpha)
        arrow(geom.sdid_post - 1.96 * geom.se, ci_alpha)


def synthdid_units_plot(
    estimates: Estimates,
    negligible_threshold: float = 0.001,
    negligible_alpha: float = 0.3,
    se_method: Optional[str] = "jackknife",
    se_replications: int = 200,
    units: Optional[Sequence] = None,
    ax: Optional[Any] = None,
    figsize: Optional[tuple[float, float]] = None,
):
    """Unit-by-unit difference-in-differences, with dot size showing ``omega``.

    Each dot is one control unit's own diff-in-diff against the treated
    average; the horizontal lines mark the estimate and a 95% interval.  Units
    with negligible weight are drawn as small transparent crosses.
    """
    plt = _require_matplotlib()
    named = _as_named(estimates)
    n = len(named)

    if ax is None:
        if figsize is None:
            widest = max(est.setup.N0 for est in named.values())
            figsize = (max(7.0, 0.22 * widest) * n, 4.5)
        fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
        axes = axes.ravel()
    else:
        axes = np.atleast_1d(np.asarray(ax, dtype=object)).ravel()
        fig = axes[0].figure

    for panel, (name, estimate) in enumerate(named.items()):
        axis = axes[panel]
        setup = estimate.setup
        w = estimate.weights
        Y = setup.Y - contract3(setup.X, w.beta)
        N0, T0, N1, T1 = setup.N0, setup.T0, setup.N1, setup.T1

        lambda_pre = np.concatenate([w.lambda_, np.zeros(T1)])
        lambda_post = np.concatenate([np.zeros(T0), np.full(T1, 1.0 / T1)])
        omega_treat = np.concatenate([np.zeros(N0), np.full(N1, 1.0 / N1)])
        contrast = lambda_post - lambda_pre
        difs = float(omega_treat @ Y @ contrast) - Y[:N0] @ contrast

        labels = np.asarray(setup.unit_labels()[:N0])
        keep = np.ones(N0, dtype=bool) if units is None else np.isin(labels, np.asarray(units))
        weight = np.asarray(w.omega)[keep]
        y = difs[keep]
        x = np.arange(keep.sum())

        big = weight > negligible_threshold
        axis.scatter(x[big], y[big], s=400 * weight[big] + 8, color="black", zorder=3)
        axis.scatter(
            x[~big], y[~big], marker="x", s=16, color="black", alpha=negligible_alpha, zorder=2
        )
        axis.axhline(float(estimate), color="black", linewidth=1.2, zorder=1)
        if se_method not in (None, "none"):
            kwargs = {} if se_method == "jackknife" else {"replications": se_replications}
            se = estimate.se(method=se_method, **kwargs)
            if np.isfinite(se):
                axis.axhline(float(estimate) - 1.96 * se, color="black", linewidth=0.8, alpha=0.5)
                axis.axhline(float(estimate) + 1.96 * se, color="black", linewidth=0.8, alpha=0.5)
        axis.set_xticks(x)
        axis.set_xticklabels(labels[keep], rotation=90, fontsize=7)
        if n > 1:
            axis.set_title(name, fontsize=10)
        axis.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig, (axes[0] if n == 1 else axes)


def synthdid_rmse_plot(
    estimates: Estimates,
    ax: Optional[Any] = None,
    figsize: tuple[float, float] = (7.0, 4.0),
):
    """Solver diagnostic: penalized RMSE against Frank-Wolfe iteration, on a log scale."""
    plt = _require_matplotlib()
    named = _as_named(estimates)
    if ax is None:
        fig, axis = plt.subplots(figsize=figsize)
    else:
        axis, fig = ax, ax.figure

    for i, (name, estimate) in enumerate(named.items()):
        vals = estimate.weights.vals
        if vals is None or len(vals) == 0:
            continue
        rmse = np.sqrt(np.asarray(vals, dtype=float))
        axis.plot(
            np.arange(1, len(rmse) + 1),
            rmse,
            label=name,
            color=_ESTIMATE_COLORS[i % len(_ESTIMATE_COLORS)],
        )
    axis.set_yscale("log")
    axis.set_xlabel("iteration")
    axis.set_ylabel("rmse")
    axis.legend(loc="upper right", frameon=False, fontsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, axis


def synthdid_placebo_plot(
    estimate: SynthDIDEstimate,
    overlay: bool = False,
    treated_fraction: Optional[float] = None,
    **kwargs,
):
    """Plot the estimate beside a placebo fit that uses pre-treatment data only.

    A placebo effect far from zero is evidence against the identifying
    assumptions.
    """
    estimates = {
        "estimate": estimate,
        "placebo": synthdid_placebo(estimate, treated_fraction=treated_fraction),
    }
    facet = ["estimate", "estimate"] if overlay else None
    return synthdid_plot(estimates, facet=facet, **kwargs)
