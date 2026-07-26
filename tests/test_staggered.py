"""Staggered adoption."""

import numpy as np
import pandas as pd
import pytest

from synthdid import load_california_prop99, synthdid_estimate
from synthdid.staggered import (
    staggered_panel_matrices,
    staggered_synthdid_estimate,
)


def make_staggered_panel(seed=0, n_control=25, cohorts=((6, 3), (10, 4)), T=16, tau=2.0):
    """A synthetic panel where every treated cohort has the same true effect ``tau``."""
    rng = np.random.default_rng(seed)
    n_treated = sum(size for _, size in cohorts)
    n = n_control + n_treated
    unit_effect = rng.normal(scale=3.0, size=n)
    time_effect = np.linspace(0, 5, T)
    Y = unit_effect[:, None] + time_effect[None, :] + rng.normal(scale=0.05, size=(n, T))

    adoption = np.full(n, -1)
    row = n_control
    for start, size in cohorts:
        adoption[row : row + size] = start
        Y[row : row + size, start:] += tau
        row += size

    records = []
    for i in range(n):
        for t in range(T):
            treated = int(adoption[i] >= 0 and t >= adoption[i])
            records.append((f"unit{i:03d}", 2000 + t, Y[i, t], treated))
    return pd.DataFrame(records, columns=["unit", "year", "y", "d"])


@pytest.fixture(scope="module")
def staggered_df():
    return make_staggered_panel()


def test_reshapes_and_labels_the_cohorts(staggered_df):
    panel = staggered_panel_matrices(staggered_df, "unit", "year", "y", "d")
    assert panel.Y.shape == (32, 16)
    assert int(panel.never_treated.sum()) == 25
    np.testing.assert_array_equal(panel.adoption_times, [6, 10])
    # never-treated units come first, then cohorts in adoption order
    assert np.all(np.diff(panel.adoption) >= 0)


def test_single_adoption_date_reproduces_the_block_estimator(prop99):
    df = load_california_prop99()
    staggered = staggered_synthdid_estimate(
        df, unit="State", time="Year", outcome="PacksPerCapita", treatment="treated"
    )
    block = synthdid_estimate(prop99.Y, prop99.N0, prop99.T0)
    assert staggered.att == pytest.approx(float(block), rel=1e-12)
    assert len(staggered.by_cohort) == 1


def test_recovers_a_common_effect_across_cohorts(staggered_df):
    est = staggered_synthdid_estimate(staggered_df, unit="unit", time="year", outcome="y", treatment="d")
    assert est.att == pytest.approx(2.0, abs=0.1)
    assert len(est.by_cohort) == 2
    for tau in est.by_cohort["tau"]:
        assert tau == pytest.approx(2.0, abs=0.15)


def test_cohort_weights_are_treated_unit_periods(staggered_df):
    est = staggered_synthdid_estimate(staggered_df, unit="unit", time="year", outcome="y", treatment="d")
    by = est.by_cohort
    raw = by["n_treated"] * by["n_post"]
    np.testing.assert_allclose(by["weight"], raw / raw.sum())
    assert est.att == pytest.approx(float((by["weight"] * by["tau"]).sum()))


def test_not_yet_treated_controls_truncate_the_post_window(staggered_df):
    est = staggered_synthdid_estimate(
        staggered_df, unit="unit", time="year", outcome="y", treatment="d",
        control_pool="not_yet_treated",
    )
    first = est.by_cohort.iloc[0]
    # the first cohort's window now stops where the second cohort adopts
    assert first["n_post"] == 4
    assert first["n_control"] == 25 + 4  # never treated, plus the later cohort
    assert est.att == pytest.approx(2.0, abs=0.15)


@pytest.mark.parametrize("estimator", ["synthdid", "sc", "did"])
def test_every_block_estimator_works(staggered_df, estimator):
    est = staggered_synthdid_estimate(
        staggered_df, unit="unit", time="year", outcome="y", treatment="d", estimator=estimator
    )
    assert np.isfinite(est.att)


@pytest.mark.parametrize("method", ["jackknife", "bootstrap", "placebo"])
def test_standard_errors(staggered_df, method):
    est = staggered_synthdid_estimate(staggered_df, unit="unit", time="year", outcome="y", treatment="d")
    se = est.se(method=method, replications=10, random_state=0)
    assert np.isfinite(se) and se > 0
    low, high = est.ci(method=method, replications=10, random_state=0)
    assert low < est.att < high


def test_non_absorbing_treatment_is_rejected(staggered_df):
    broken = staggered_df.copy()
    mask = (broken["unit"] == "unit025") & (broken["year"] == 2010)
    broken.loc[mask, "d"] = 0
    with pytest.raises(ValueError, match="not absorbing"):
        staggered_panel_matrices(broken, "unit", "year", "y", "d")


def test_unit_treated_from_the_first_period_is_rejected(staggered_df):
    broken = staggered_df.copy()
    broken.loc[broken["unit"] == "unit025", "d"] = 1
    with pytest.raises(ValueError, match="no pre-treatment data"):
        staggered_panel_matrices(broken, "unit", "year", "y", "d")


def test_bad_control_pool_is_rejected(staggered_df):
    with pytest.raises(ValueError, match="control_pool"):
        staggered_synthdid_estimate(
            staggered_df, unit="unit", time="year", outcome="y", treatment="d",
            control_pool="whatever",
        )
