"""Solver internals: Frank-Wolfe steps, the covariate solver and sparsification."""

import numpy as np
import pytest

from synthdid.solver import (
    contract3,
    fw_step,
    sc_weight_fw,
    sc_weight_fw_covariates,
    sparsify_function,
)


def test_contract3_sums_covariate_slices():
    X = np.arange(24, dtype=float).reshape(2, 3, 4)
    v = np.array([1.0, 0.0, -2.0, 0.5])
    np.testing.assert_allclose(contract3(X, v), np.einsum("ntc,c->nt", X, v))


def test_contract3_with_no_covariates_is_zero():
    X = np.zeros((3, 4, 0))
    np.testing.assert_array_equal(contract3(X, None), np.zeros((3, 4)))


def test_fw_step_stays_on_the_simplex():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(20, 6))
    b = rng.normal(size=20)
    x = np.full(6, 1 / 6)
    for _ in range(50):
        x = fw_step(A, x, b, eta=0.1)
        assert x.min() >= -1e-12
        assert x.sum() == pytest.approx(1.0)


def test_fw_step_decreases_the_objective():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(30, 5))
    b = A @ np.array([0.5, 0.5, 0.0, 0.0, 0.0]) + 0.01 * rng.normal(size=30)
    eta = 0.05

    def objective(x):
        r = A @ x - b
        return r @ r + eta * (x @ x)

    x = np.full(5, 0.2)
    previous = objective(x)
    for _ in range(20):
        x = fw_step(A, x, b, eta)
        current = objective(x)
        assert current <= previous + 1e-12
        previous = current


def test_sc_weight_fw_recovers_a_known_convex_combination():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(60, 4))
    truth = np.array([0.6, 0.4, 0.0, 0.0])
    Y = np.column_stack([A, A @ truth])
    out = sc_weight_fw(Y, zeta=0.0, intercept=False, min_decrease=1e-10, max_iter=20_000)
    np.testing.assert_allclose(out["lambda"], truth, atol=5e-3)


def test_sc_weight_fw_objective_is_monotone():
    rng = np.random.default_rng(3)
    Y = rng.normal(size=(25, 8))
    vals = sc_weight_fw(Y, zeta=0.1, min_decrease=1e-9, max_iter=500)["vals"]
    assert np.all(np.diff(vals) <= 1e-12)


def test_sc_weight_fw_intercept_absorbs_a_level_shift():
    rng = np.random.default_rng(4)
    Y = rng.normal(size=(20, 5))
    base = sc_weight_fw(Y, zeta=0.05, intercept=True, min_decrease=1e-9, max_iter=2000)
    shifted = sc_weight_fw(
        Y + 100.0, zeta=0.05, intercept=True, min_decrease=1e-9, max_iter=2000
    )
    np.testing.assert_allclose(base["lambda"], shifted["lambda"], atol=1e-8)


def test_covariate_solver_recovers_the_coefficient():
    rng = np.random.default_rng(5)
    Y = rng.normal(size=(9, 7))
    X = rng.normal(size=(9, 7, 1))
    beta_true = 2.5
    out = sc_weight_fw_covariates(
        Y + beta_true * X[:, :, 0], X, min_decrease=1e-9, max_iter=5000
    )
    assert out["beta"][0] == pytest.approx(beta_true, abs=0.2)
    assert out["lambda"].sum() == pytest.approx(1.0)
    assert out["omega"].sum() == pytest.approx(1.0)


def test_covariate_solver_can_hold_weights_fixed():
    rng = np.random.default_rng(6)
    Y = rng.normal(size=(6, 5))
    X = rng.normal(size=(6, 5, 1))
    lam = np.array([0.4, 0.3, 0.2, 0.1])
    omega = np.array([0.5, 0.25, 0.15, 0.1, 0.0])
    out = sc_weight_fw_covariates(
        Y, X, lambda_=lam, omega=omega, update_lambda=False, update_omega=False, max_iter=50
    )
    np.testing.assert_allclose(out["lambda"], lam)
    np.testing.assert_allclose(out["omega"], omega)


def test_sparsify_zeroes_small_weights_and_renormalizes():
    v = np.array([1.0, 0.2, 0.1, 0.05])
    out = sparsify_function(v)
    assert out.sum() == pytest.approx(1.0)
    assert out[0] == 1.0  # everything at or below max/4 is dropped
    assert np.all(out[1:] == 0)


def test_sparsify_keeps_weights_above_the_threshold():
    v = np.array([1.0, 0.5, 0.05])
    out = sparsify_function(v)
    np.testing.assert_allclose(out, [2 / 3, 1 / 3, 0.0])
