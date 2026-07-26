import numpy as np
import pytest

from synthdid import load_california_prop99, panel_matrices, random_low_rank


@pytest.fixture(scope="session")
def prop99():
    """The California Proposition 99 panel, reshaped."""
    return panel_matrices(load_california_prop99())


@pytest.fixture(scope="session")
def low_rank():
    """A synthetic low-rank panel with a known treatment effect of 1."""
    return random_low_rank(n_0=30, n_1=5, T_0=25, T_1=8, random_state=20240101)


@pytest.fixture(scope="session")
def small_panel():
    """A small panel, so the resampling variance estimators stay fast."""
    return random_low_rank(n_0=12, n_1=3, T_0=10, T_1=4, random_state=7)


def assert_close(actual, expected, rtol=1e-7, atol=0.0):
    np.testing.assert_allclose(np.asarray(actual, dtype=float), expected, rtol=rtol, atol=atol)
