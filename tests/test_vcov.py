"""The three variance estimators."""

import numpy as np
import pytest

from synthdid import (
    bootstrap_se,
    did_estimate,
    jackknife_se,
    placebo_se,
    sc_estimate,
    synthdid_estimate,
    synthdid_se,
    vcov,
)

METHODS = ["bootstrap", "jackknife", "placebo"]


@pytest.fixture(scope="module")
def estimate(small_panel):
    return synthdid_estimate(small_panel["Y"], small_panel["N0"], small_panel["T0"])


@pytest.mark.parametrize("method", METHODS)
def test_every_method_returns_a_positive_standard_error(estimate, method):
    se = synthdid_se(estimate, method=method, replications=20, random_state=0)
    assert np.isfinite(se)
    assert se > 0


def test_vcov_returns_a_one_by_one_matrix(estimate):
    v = vcov(estimate, method="jackknife")
    assert v.shape == (1, 1)
    assert v[0, 0] == pytest.approx(jackknife_se(estimate) ** 2)


@pytest.mark.parametrize("method", METHODS)
def test_resampling_is_reproducible(estimate, method):
    kwargs = dict(method=method, replications=15, random_state=42)
    assert synthdid_se(estimate, **kwargs) == synthdid_se(estimate, **kwargs)


@pytest.mark.parametrize("method", METHODS)
def test_standard_errors_scale_with_the_outcome(small_panel, method):
    Y, N0, T0 = small_panel["Y"], small_panel["N0"], small_panel["T0"]
    scale = 1000.0
    base = synthdid_se(
        synthdid_estimate(Y, N0, T0), method=method, replications=15, random_state=3
    )
    scaled = synthdid_se(
        synthdid_estimate(scale * Y, N0, T0), method=method, replications=15, random_state=3
    )
    assert scaled == pytest.approx(scale * base, rel=1e-6)


def test_one_treated_unit_defeats_bootstrap_and_jackknife(prop99):
    est = synthdid_estimate(prop99.Y, prop99.N0, prop99.T0)
    assert np.isnan(jackknife_se(est))
    assert np.isnan(bootstrap_se(est, 10, random_state=0))
    # the placebo is the method the paper recommends in this case
    assert placebo_se(est, 20, random_state=0) > 0


def test_placebo_needs_more_controls_than_treated(small_panel):
    Y, N0, T0 = small_panel["Y"], small_panel["N0"], small_panel["T0"]
    lopsided = synthdid_estimate(Y, 2, T0)
    with pytest.raises(ValueError, match="more controls than treated"):
        placebo_se(lopsided, 5, random_state=0)


def test_fixed_weight_jackknife_returns_nan_for_a_single_weighted_control(small_panel):
    Y, N0, T0 = small_panel["Y"], small_panel["N0"], small_panel["T0"]
    est = synthdid_estimate(Y, N0, T0)
    est.weights.omega = np.eye(N0)[0]
    assert np.isnan(jackknife_se(est))


@pytest.mark.parametrize("estimator", [synthdid_estimate, sc_estimate, did_estimate])
def test_works_for_every_estimator(small_panel, estimator):
    est = estimator(small_panel["Y"], small_panel["N0"], small_panel["T0"])
    se = synthdid_se(est, method="placebo", replications=15, random_state=1)
    assert np.isfinite(se) and se > 0


def test_unknown_method_is_rejected(estimate):
    with pytest.raises(ValueError, match="method must be one of"):
        vcov(estimate, method="magic")
