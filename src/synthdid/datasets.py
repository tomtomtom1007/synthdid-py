"""Bundled example panels, identical to the data shipped with the R package."""

from __future__ import annotations

from importlib import resources

import pandas as pd

__all__ = [
    "load_california_prop99",
    "load_cps",
    "load_penn",
    "available_datasets",
]

_FILES = {
    "california_prop99": "california_prop99.csv",
    "cps": "CPS.csv",
    "penn": "PENN.csv",
}


def _read(name: str) -> pd.DataFrame:
    path = resources.files("synthdid.data").joinpath(_FILES[name])
    with resources.as_file(path) as file:
        return pd.read_csv(
            file, sep=";", true_values=["TRUE"], false_values=["FALSE"]
        )


def available_datasets() -> list[str]:
    """Names accepted by the loaders in this module."""
    return sorted(_FILES)


def load_california_prop99() -> pd.DataFrame:
    """Per-capita cigarette sales by US state, 1970-2000 (Abadie et al. 2010).

    California raised its tobacco tax in 1989 (Proposition 99); ``treated`` is 1
    for California from 1989 on.  This is the canonical synthetic-control
    example and the one used throughout the synthdid paper.

    Returns
    -------
    DataFrame with columns ``State``, ``Year``, ``PacksPerCapita``, ``treated``
    (1209 rows: 39 states x 31 years).

    References
    ----------
    Abadie, Diamond and Hainmueller (2010), *Synthetic Control Methods for
    Comparative Case Studies*, JASA 105(490), 493-505.
    """
    return _read("california_prop99")


def load_cps() -> pd.DataFrame:
    """State-year CPS extract with wages, hours and policy indicators.

    Columns: ``state``, ``year``, ``log_wage``, ``hours``, ``urate``,
    ``min_wage``, ``open_carry``, ``abort_ban``.
    """
    return _read("cps")


def load_penn() -> pd.DataFrame:
    """Penn World Table extract: country-year log GDP with democracy indicators.

    Columns: ``country``, ``year``, ``log_gdp``, ``dem``, ``educ``.  Used in the
    R vignette to demonstrate covariate adjustment.
    """
    return _read("penn")
