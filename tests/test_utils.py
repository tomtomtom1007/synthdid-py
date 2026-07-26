"""Reshaping a long panel, and the small numerical helpers."""

import numpy as np
import pandas as pd
import pytest

from synthdid import collapsed_form, load_california_prop99, panel_matrices
from synthdid.utils import pairwise_sum_decreasing, sum_normalize


@pytest.fixture
def panel():
    return load_california_prop99()


def test_reshapes_prop99(panel, prop99):
    assert prop99.Y.shape == (39, 31)
    assert prop99.N0 == 38
    assert prop99.T0 == 19
    assert prop99.N1 == 1
    assert prop99.T1 == 12
    assert prop99.units[-1] == "California"
    assert prop99.time[0] == 1970
    assert prop99.W[-1, prop99.T0] == 1
    assert prop99.W[:-1].sum() == 0


def test_row_order_of_the_input_does_not_matter(panel):
    shuffled = panel.sample(frac=1.0, random_state=0).reset_index(drop=True)
    expected = panel_matrices(panel)
    actual = panel_matrices(shuffled)
    np.testing.assert_array_equal(actual.Y, expected.Y)
    np.testing.assert_array_equal(actual.units, expected.units)
    assert (actual.N0, actual.T0) == (expected.N0, expected.T0)


def test_columns_can_be_named_or_positional(panel, prop99):
    by_name = panel_matrices(
        panel, unit="State", time="Year", outcome=3, treatment="treated"
    )
    np.testing.assert_array_equal(by_name.Y, prop99.Y)


def test_extra_columns_are_ignored(panel, prop99):
    rng = np.random.default_rng(0)
    extra = pd.DataFrame(rng.random((len(panel), 3)), columns=["a", "b", "c"])
    augmented = pd.concat([extra, panel], axis=1)
    out = panel_matrices(
        augmented, unit="State", time="Year", outcome="PacksPerCapita", treatment="treated"
    )
    np.testing.assert_array_equal(out.Y, prop99.Y)


def test_covariates_are_reshaped_alongside(panel):
    rng = np.random.default_rng(1)
    augmented = panel.assign(z=rng.random(len(panel)))
    out = panel_matrices(
        augmented,
        unit="State",
        time="Year",
        outcome="PacksPerCapita",
        treatment="treated",
        covariates=["z"],
    )
    assert out.X.shape == (39, 31, 1)
    assert not np.isnan(out.X).any()


def test_unbalanced_panel_is_rejected(panel):
    with pytest.raises(ValueError, match="balanced panel"):
        panel_matrices(panel.drop(index=10))
    with pytest.raises(ValueError, match="balanced panel"):
        panel_matrices(pd.concat([panel, panel.iloc[5:10]], ignore_index=True))


def test_missing_values_are_rejected(panel):
    broken = panel.copy()
    broken.loc[0, "PacksPerCapita"] = np.nan
    with pytest.raises(ValueError, match="Missing values"):
        panel_matrices(broken)


def test_non_binary_treatment_is_rejected(panel):
    broken = panel.copy()
    broken.loc[0, "treated"] = 2
    with pytest.raises(ValueError, match="0 or 1"):
        panel_matrices(broken)


def test_constant_treatment_is_rejected(panel):
    broken = panel.assign(treated=0)
    with pytest.raises(ValueError, match="no variation"):
        panel_matrices(broken)


def test_a_second_treated_unit_sorts_last(panel):
    modified = panel.copy()
    mask = (modified["State"] == "Kansas") & (modified["Year"] >= 1989)
    modified.loc[mask, "treated"] = 1
    out = panel_matrices(modified)
    assert list(out.units[-2:]) == ["California", "Kansas"]
    assert out.N0 == 37


def test_non_simultaneous_adoption_is_rejected(panel):
    modified = panel.copy()
    mask = (modified["State"] == "Kansas") & (modified["Year"] >= 1988)
    modified.loc[mask, "treated"] = 1
    with pytest.raises(ValueError, match="not simultaneous"):
        panel_matrices(modified)


def test_treated_last_can_be_switched_off(panel):
    out = panel_matrices(panel, treated_last=False)
    assert list(out.units) == sorted(panel["State"].unique())


def test_collapsed_form_averages_the_treated_block():
    Y = np.arange(20, dtype=float).reshape(4, 5)
    collapsed = collapsed_form(Y, N0=3, T0=3)
    assert collapsed.shape == (4, 4)
    np.testing.assert_allclose(collapsed[:3, :3], Y[:3, :3])
    np.testing.assert_allclose(collapsed[:3, 3], Y[:3, 3:].mean(axis=1))
    np.testing.assert_allclose(collapsed[3, :3], Y[3:, :3].mean(axis=0))
    assert collapsed[3, 3] == pytest.approx(Y[3:, 3:].mean())


def test_sum_normalize_handles_all_zero_input():
    np.testing.assert_allclose(sum_normalize(np.array([1.0, 3.0])), [0.25, 0.75])
    np.testing.assert_allclose(sum_normalize(np.zeros(4)), 0.25)


def test_pairwise_sum_extends_the_shorter_trace_with_its_minimum():
    x = np.array([10.0, 5.0, 2.0])
    y = np.array([8.0, 3.0])
    np.testing.assert_allclose(pairwise_sum_decreasing(x, y), [18.0, 8.0, 5.0])


def test_panel_data_unpacks_like_the_r_list(prop99):
    Y, N0, T0, W = prop99
    assert Y.shape == prop99.Y.shape
    assert (N0, T0) == (prop99.N0, prop99.T0)
    assert W.shape == prop99.W.shape
