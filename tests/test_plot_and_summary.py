"""Plots render, and summaries report what they claim to."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from synthdid import (  # noqa: E402
    did_estimate,
    sc_estimate,
    synthdid_controls,
    synthdid_estimate,
    synthdid_placebo_plot,
    synthdid_plot,
    synthdid_rmse_plot,
    synthdid_units_plot,
    summary,
)


@pytest.fixture(scope="module")
def estimates(prop99):
    Y, N0, T0 = prop99.Y, prop99.N0, prop99.T0
    labels = dict(unit_names=prop99.units, time_labels=prop99.time)
    return {
        "Diff-in-Diff": did_estimate(Y, N0, T0, **labels),
        "Synthetic Control": sc_estimate(Y, N0, T0, **labels),
        "Synthetic Diff-in-Diff": synthdid_estimate(Y, N0, T0, **labels),
    }


@pytest.fixture(scope="module")
def sdid(estimates):
    return estimates["Synthetic Diff-in-Diff"]


def test_plot_renders(sdid):
    fig, ax = synthdid_plot(sdid, se_method="none")
    assert ax.lines
    matplotlib.pyplot.close(fig)


def test_plot_with_confidence_arrows(sdid):
    fig, ax = synthdid_plot(sdid, se_method="placebo", se_replications=5)
    assert len(ax.patches) >= 3  # effect arrow plus two interval arrows
    matplotlib.pyplot.close(fig)


def test_plot_of_several_estimates_uses_one_panel_each(estimates):
    fig, axes = synthdid_plot(estimates, se_method="none")
    assert len(axes) == 3
    matplotlib.pyplot.close(fig)


def test_facet_overlays_estimates_in_one_panel(estimates):
    fig, ax = synthdid_plot(estimates, facet=["all"] * 3, se_method="none")
    assert not isinstance(ax, np.ndarray)
    matplotlib.pyplot.close(fig)


def test_overlay_suppresses_the_diagram(sdid):
    fig, ax = synthdid_plot(sdid, overlay=1.0, se_method="none")
    matplotlib.pyplot.close(fig)


def test_spaghetti_units_are_drawn(sdid):
    top = list(sdid.controls().index[:5])
    fig, ax = synthdid_plot(sdid, spaghetti_units=top, se_method="none")
    assert len(ax.texts) == 5
    matplotlib.pyplot.close(fig)


def test_units_plot_renders(sdid, prop99):
    fig, ax = synthdid_units_plot(sdid, se_method="none")
    assert len(ax.get_xticklabels()) == prop99.N0
    matplotlib.pyplot.close(fig)


def test_units_plot_can_restrict_to_named_units(sdid):
    fig, ax = synthdid_units_plot(sdid, units=["Nevada", "Utah"], se_method="none")
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Nevada", "Utah"]
    matplotlib.pyplot.close(fig)


def test_rmse_plot_renders(sdid):
    fig, ax = synthdid_rmse_plot(sdid)
    assert ax.get_yscale() == "log"
    matplotlib.pyplot.close(fig)


def test_placebo_plot_renders(sdid):
    fig, axes = synthdid_placebo_plot(sdid, se_method="none")
    assert len(axes) == 2
    matplotlib.pyplot.close(fig)


def test_controls_table_is_sorted_and_truncated(sdid):
    table = synthdid_controls(sdid, mass=0.9)
    assert table.index[0] == "Nevada"
    assert table.is_monotonic_decreasing
    assert table.sum() >= 0.9
    assert len(table) < len(sdid.weights.omega)


def test_controls_table_for_several_estimates(estimates):
    table = synthdid_controls(estimates)
    assert list(table.columns) == list(estimates)


def test_lambda_controls_table_lists_periods(sdid):
    table = synthdid_controls(sdid, weight_type="lambda")
    assert table.index[0] == 1988
    assert table.sum() >= 0.9


def test_summary_reports_dimensions_and_weights(sdid, prop99):
    info = summary(sdid, se_method="placebo", replications=5, random_state=0)
    assert info.estimate == pytest.approx(float(sdid))
    assert info.dimensions["N0"] == prop99.N0
    assert info.dimensions["T1"] == prop99.T1
    assert 1 < info.dimensions["N0_effective"] < prop99.N0
    assert np.isfinite(info.se)
    assert "synthdid" in repr(info)


def test_summary_of_a_synthetic_control_has_no_effective_pre_periods(prop99):
    est = sc_estimate(prop99.Y, prop99.N0, prop99.T0)
    info = summary(est, se_method="placebo", replications=5, random_state=0)
    assert np.isinf(info.dimensions["T0_effective"])


def test_repr_is_cheap_and_informative(sdid):
    text = repr(sdid)
    assert "synthdid_estimate" in text
    assert "-15.60" in text


def test_format_matches_the_r_one_liner(prop99):
    est = synthdid_estimate(prop99.Y, prop99.N0, prop99.T0)
    text = est.format(se_method="placebo")
    assert text.startswith("synthdid_estimate: -15.604 +- ")
    assert "Effective N0/N0" in text
