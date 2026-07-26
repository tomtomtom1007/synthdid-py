"""Synthetic Difference-in-Differences for Python.

A complete port of the reference R package
`synthdid <https://github.com/synth-inference/synthdid>`_ accompanying
Arkhangelsky, Athey, Hirshberg, Imbens and Wager (2021), *Synthetic
Difference-in-Differences*, American Economic Review 111(12), 4088-4118.

Quick start
-----------
>>> from synthdid import load_california_prop99, panel_matrices, synthdid_estimate
>>> panel = panel_matrices(load_california_prop99())
>>> tau = synthdid_estimate(panel.Y, panel.N0, panel.T0)
>>> round(float(tau), 3)
-15.604
"""

from .datasets import (
    available_datasets,
    load_california_prop99,
    load_cps,
    load_penn,
)
from .estimate import (
    Setup,
    SynthDIDEstimate,
    Weights,
    did_estimate,
    refit,
    sc_estimate,
    synthdid_effect_curve,
    synthdid_estimate,
    synthdid_placebo,
)
from .plot import (
    synthdid_placebo_plot,
    synthdid_plot,
    synthdid_rmse_plot,
    synthdid_units_plot,
)
from .solver import (
    contract3,
    fw_step,
    sc_weight_fw,
    sc_weight_fw_covariates,
    sparsify_function,
)
from .staggered import StaggeredEstimate, staggered_synthdid_estimate
from .summary import format_estimate, summary, synthdid_controls
from .utils import (
    PanelData,
    collapsed_form,
    panel_matrices,
    random_low_rank,
)
from .vcov import (
    bootstrap_se,
    jackknife_se,
    placebo_se,
    synthdid_se,
    vcov,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # estimators
    "synthdid_estimate",
    "sc_estimate",
    "did_estimate",
    "staggered_synthdid_estimate",
    "StaggeredEstimate",
    "SynthDIDEstimate",
    "Setup",
    "Weights",
    "refit",
    # inference
    "vcov",
    "synthdid_se",
    "bootstrap_se",
    "jackknife_se",
    "placebo_se",
    # diagnostics
    "summary",
    "synthdid_controls",
    "format_estimate",
    "synthdid_placebo",
    "synthdid_effect_curve",
    # plots
    "synthdid_plot",
    "synthdid_units_plot",
    "synthdid_rmse_plot",
    "synthdid_placebo_plot",
    # data handling
    "panel_matrices",
    "PanelData",
    "collapsed_form",
    "random_low_rank",
    # solver internals
    "contract3",
    "fw_step",
    "sc_weight_fw",
    "sc_weight_fw_covariates",
    "sparsify_function",
    # datasets
    "load_california_prop99",
    "load_cps",
    "load_penn",
    "available_datasets",
]
