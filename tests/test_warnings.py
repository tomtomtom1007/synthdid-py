"""The public API must not emit numerical warnings on any supported platform.

numpy's ``@`` operator raises spurious "divide by zero", "overflow" and
"invalid value" RuntimeWarnings on macOS builds of numpy 2.0 linked against
Apple Accelerate -- the values are correct, only the floating-point flags are
wrong.  Python 3.9 and 3.10 on macOS pin numpy to that range, so users there hit
it on every call.  The package therefore uses ``ndarray.dot`` for matrix
products; these tests fail loudly if a ``@`` creeps back into a hot path.
"""

import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import synthdid as sd  # noqa: E402


@pytest.fixture(autouse=True)
def numerical_warnings_are_errors():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


def test_estimators_are_quiet(prop99):
    labels = dict(unit_names=prop99.units, time_labels=prop99.time)
    sd.synthdid_estimate(prop99.Y, prop99.N0, prop99.T0, **labels)
    sd.sc_estimate(prop99.Y, prop99.N0, prop99.T0, **labels)
    sd.did_estimate(prop99.Y, prop99.N0, prop99.T0, **labels)


def test_diagnostics_are_quiet(prop99):
    est = sd.synthdid_estimate(
        prop99.Y, prop99.N0, prop99.T0, unit_names=prop99.units, time_labels=prop99.time
    )
    est.effect_curve()
    est.placebo()
    est.controls()
    est.summary(se_method="placebo", replications=5, random_state=0)


def test_variance_estimators_are_quiet(small_panel):
    est = sd.synthdid_estimate(small_panel["Y"], small_panel["N0"], small_panel["T0"])
    est.se(method="jackknife")
    est.se(method="bootstrap", replications=5, random_state=0)
    est.se(method="placebo", replications=5, random_state=0)


def test_covariate_adjustment_is_quiet(small_panel):
    X = (small_panel["Y"] - small_panel["L"])[:, :, None]
    sd.synthdid_estimate(small_panel["Y"], small_panel["N0"], small_panel["T0"], X=X)


def test_plots_are_quiet(prop99):
    labels = dict(unit_names=prop99.units, time_labels=prop99.time)
    est = sd.synthdid_estimate(prop99.Y, prop99.N0, prop99.T0, **labels)
    estimates = {
        "DiD": sd.did_estimate(prop99.Y, prop99.N0, prop99.T0, **labels),
        "SDID": est,
    }
    for figure, _ in (
        est.plot(se_method="none"),
        est.plot(overlay=1.0, se_method="none"),
        est.units_plot(se_method="none"),
        est.rmse_plot(),
        est.placebo_plot(se_method="none"),
        sd.synthdid_plot(estimates, se_method="none"),
        sd.synthdid_plot(estimates, facet=["a", "a"], se_method="none"),
    ):
        matplotlib.pyplot.close(figure)


def test_staggered_is_quiet():
    sd.staggered_synthdid_estimate(
        sd.load_california_prop99(),
        unit="State",
        time="Year",
        outcome="PacksPerCapita",
        treatment="treated",
    )


def test_random_low_rank_is_quiet():
    sd.random_low_rank(n_0=10, n_1=2, T_0=8, T_1=3, random_state=0)


def test_the_guard_itself_works():
    """Sanity check: a genuine numerical warning would still be caught."""
    with pytest.raises(RuntimeWarning):
        np.array([1.0]) / np.array([0.0])
