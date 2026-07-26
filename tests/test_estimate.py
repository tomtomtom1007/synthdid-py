"""Point-estimate correctness, including the invariances the R package tests for."""

import numpy as np
import pandas as pd
import pytest

from synthdid import (
    did_estimate,
    sc_estimate,
    synthdid_effect_curve,
    synthdid_estimate,
    synthdid_placebo,
)

# Values produced by the reference R package (synthdid 0.0.9) on
# california_prop99, to five decimal places.
R_SYNTHDID = -15.60383
R_SC = -19.61966
R_DID = -27.34911


def test_prop99_matches_r_reference(prop99):
    assert synthdid_estimate(prop99.Y, prop99.N0, prop99.T0) == pytest.approx(R_SYNTHDID, abs=1e-4)
    assert sc_estimate(prop99.Y, prop99.N0, prop99.T0) == pytest.approx(R_SC, abs=1e-4)
    assert did_estimate(prop99.Y, prop99.N0, prop99.T0) == pytest.approx(R_DID, abs=1e-4)


def test_did_is_the_textbook_two_by_two(prop99):
    Y, N0, T0 = prop99.Y, prop99.N0, prop99.T0
    manual = (
        Y[N0:, T0:].mean()
        - Y[N0:, :T0].mean()
        - Y[:N0, T0:].mean()
        + Y[:N0, :T0].mean()
    )
    assert did_estimate(Y, N0, T0) == pytest.approx(manual, rel=1e-12)


def test_weights_live_on_the_simplex(prop99):
    est = synthdid_estimate(prop99.Y, prop99.N0, prop99.T0)
    for w in (est.weights.omega, est.weights.lambda_):
        assert w.min() >= 0
        assert w.sum() == pytest.approx(1.0)
    assert len(est.weights.omega) == prop99.N0
    assert len(est.weights.lambda_) == prop99.T0


def test_sc_pins_time_weights_to_zero(prop99):
    est = sc_estimate(prop99.Y, prop99.N0, prop99.T0)
    assert np.all(est.weights.lambda_ == 0)


def test_did_uses_uniform_weights(prop99):
    est = did_estimate(prop99.Y, prop99.N0, prop99.T0)
    np.testing.assert_allclose(est.weights.omega, 1 / prop99.N0)
    np.testing.assert_allclose(est.weights.lambda_, 1 / prop99.T0)


def test_dataframe_input_carries_labels(prop99):
    frame = pd.DataFrame(prop99.Y, index=prop99.units, columns=prop99.time)
    est = synthdid_estimate(frame, prop99.N0, prop99.T0)
    assert est == pytest.approx(R_SYNTHDID, abs=1e-4)
    assert est.setup.unit_labels()[-1] == "California"
    assert est.controls().index[0] == "Nevada"


def test_effect_curve_averages_to_the_estimate(prop99):
    est = synthdid_estimate(prop99.Y, prop99.N0, prop99.T0)
    curve = synthdid_effect_curve(est)
    assert len(curve) == prop99.T1
    assert curve.mean() == pytest.approx(float(est), rel=1e-10)


def test_placebo_uses_only_pre_treatment_data(prop99):
    est = synthdid_estimate(prop99.Y, prop99.N0, prop99.T0)
    placebo = synthdid_placebo(est)
    assert placebo.setup.Y.shape[1] == prop99.T0
    assert placebo.setup.T0 == int(np.floor(prop99.T0 * prop99.T0 / prop99.Y.shape[1]))
    # a credible design has a placebo effect much smaller than the real one
    assert abs(float(placebo)) < abs(float(est))


ESTIMATORS = [synthdid_estimate, sc_estimate, did_estimate]


@pytest.mark.parametrize("estimator", ESTIMATORS, ids=lambda f: f.__name__)
def test_invariant_to_time_fixed_effects(low_rank, estimator):
    Y, N0, T0 = low_rank["Y"], low_rank["N0"], low_rank["T0"]
    shift = 2 * np.tile(np.arange(1, Y.shape[1] + 1), (Y.shape[0], 1))
    assert estimator(Y + shift, N0, T0) == pytest.approx(estimator(Y, N0, T0), abs=1e-8)


@pytest.mark.parametrize("estimator", ESTIMATORS[:1] + ESTIMATORS[2:], ids=lambda f: f.__name__)
def test_invariant_to_unit_fixed_effects(low_rank, estimator):
    # synthetic control is excluded: it has no unit fixed effects to absorb them
    Y, N0, T0 = low_rank["Y"], low_rank["N0"], low_rank["T0"]
    shift = 2.5 * np.tile(np.arange(1, Y.shape[0] + 1)[:, None], (1, Y.shape[1]))
    assert estimator(Y + shift, N0, T0) == pytest.approx(estimator(Y, N0, T0), abs=1e-8)


@pytest.mark.parametrize("estimator", ESTIMATORS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("scale", [1e-6, 1e6])
def test_equivariant_to_scaling(low_rank, estimator, scale):
    Y, N0, T0 = low_rank["Y"], low_rank["N0"], low_rank["T0"]
    base = estimator(Y, N0, T0)
    scaled = estimator(scale * Y, N0, T0)
    assert float(scaled) == pytest.approx(scale * float(base), rel=1e-8)
    np.testing.assert_allclose(scaled.weights.omega, base.weights.omega, atol=1e-10)
    np.testing.assert_allclose(scaled.weights.lambda_, base.weights.lambda_, atol=1e-10)


@pytest.mark.parametrize("shift", [1e-6, 0.25, 1e6])
def test_shifting_treated_post_block_shifts_the_effect(low_rank, shift):
    Y, N0, T0 = low_rank["Y"], low_rank["N0"], low_rank["T0"]
    for estimator in ESTIMATORS:
        shifted = Y.copy()
        shifted[N0:, T0:] += shift
        base = estimator(Y, N0, T0)
        moved = estimator(shifted, N0, T0)
        np.testing.assert_allclose(moved.weights.lambda_, base.weights.lambda_, atol=1e-10)
        np.testing.assert_allclose(moved.weights.omega, base.weights.omega, atol=1e-10)
        assert float(moved) == pytest.approx(float(base) + shift, abs=1e-6 * max(1.0, shift))


@pytest.mark.parametrize("shift", [1e-6, 0.25, 1e6])
def test_shifting_the_other_blocks_shifts_the_effect(low_rank, shift):
    Y, N0, T0 = low_rank["Y"], low_rank["N0"], low_rank["T0"]
    treated = slice(N0, None)
    control = slice(0, N0)
    cases = [
        # (rows, cols, sign, estimators): synthetic control cannot absorb unit effects
        (treated, slice(0, T0), -1, [synthdid_estimate, did_estimate]),
        (control, slice(0, T0), +1, [synthdid_estimate, did_estimate]),
        (control, slice(T0, None), -1, [synthdid_estimate, did_estimate]),
    ]
    for rows, cols, sign, estimators in cases:
        shifted = Y.copy()
        shifted[rows, cols] += shift
        for estimator in estimators:
            base = estimator(Y, N0, T0)
            moved = estimator(shifted, N0, T0)
            assert float(moved) == pytest.approx(
                float(base) + sign * shift, abs=1e-6 * max(1.0, shift)
            )


def test_covariate_adjustment_removes_the_noise_it_is_given(low_rank):
    Y, N0, T0 = low_rank["Y"], low_rank["N0"], low_rank["T0"]
    noise = Y - low_rank["L"]
    clean = synthdid_estimate(Y, N0, T0)
    noisy = synthdid_estimate(Y + noise, N0, T0)
    adjusted = synthdid_estimate(Y + noise, N0, T0, X=noise[:, :, None])
    assert abs(clean - adjusted) < abs(clean - noisy)
    assert adjusted.weights.beta.shape == (1,)


def test_rejects_a_panel_with_no_treated_units(low_rank):
    Y = low_rank["Y"]
    with pytest.raises(ValueError, match="at least one treated unit"):
        synthdid_estimate(Y, Y.shape[0], low_rank["T0"])


def test_rejects_a_panel_with_no_post_period(low_rank):
    Y = low_rank["Y"]
    with pytest.raises(ValueError, match="at least one post-treatment period"):
        synthdid_estimate(Y, low_rank["N0"], Y.shape[1])


def test_rejects_weights_of_the_wrong_length(low_rank):
    Y, N0, T0 = low_rank["Y"], low_rank["N0"], low_rank["T0"]
    with pytest.raises(ValueError, match="lambda_"):
        synthdid_estimate(Y, N0, T0, weights={"lambda": np.ones(T0 + 1) / (T0 + 1)})


def test_estimate_survives_a_round_trip_through_pickle(prop99):
    import copy
    import pickle

    est = synthdid_estimate(prop99.Y, prop99.N0, prop99.T0, unit_names=prop99.units)
    for revived in (pickle.loads(pickle.dumps(est)), copy.deepcopy(est)):
        assert float(revived) == float(est)
        assert revived.estimator == est.estimator
        np.testing.assert_allclose(revived.weights.omega, est.weights.omega)
        assert revived.setup.N0 == est.setup.N0
