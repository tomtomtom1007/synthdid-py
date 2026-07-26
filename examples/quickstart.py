"""Reproduce the California Proposition 99 analysis and Figure 1 of the paper.

Run with::

    python examples/quickstart.py

Writes figures to ``examples/figures/``.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from synthdid import (
    did_estimate,
    load_california_prop99,
    panel_matrices,
    sc_estimate,
    synthdid_estimate,
    synthdid_plot,
    synthdid_units_plot,
)

FIGURES = Path(__file__).parent / "figures"


def main() -> None:
    FIGURES.mkdir(exist_ok=True)

    panel = panel_matrices(load_california_prop99())
    print(panel)

    labels = dict(unit_names=panel.units, time_labels=panel.time)
    estimates = {
        "Diff-in-Diff": did_estimate(panel.Y, panel.N0, panel.T0, **labels),
        "Synthetic Control": sc_estimate(panel.Y, panel.N0, panel.T0, **labels),
        "Synthetic Diff-in-Diff": synthdid_estimate(panel.Y, panel.N0, panel.T0, **labels),
    }
    for name, estimate in estimates.items():
        print(f"{name:<24} {float(estimate):>10.5f}")

    tau = estimates["Synthetic Diff-in-Diff"]

    # California is the only treated unit, so the placebo standard error of
    # Section 5 is the only one available. It is known to be conservative.
    se = tau.se(method="placebo", replications=200, random_state=0)
    low, high = tau.ci(method="placebo", replications=200, random_state=0)
    print(f"\npoint estimate {float(tau):.2f}   se {se:.2f}   95% CI ({low:.2f}, {high:.2f})")

    print("\nControl units carrying the weight:")
    print(tau.controls().head(10).to_string())
    print("\nPre-treatment periods carrying the weight:")
    print(tau.controls(weight_type="lambda").to_string())

    print("\nEffect by post-treatment year:")
    for year, effect in zip(panel.time[panel.T0:], tau.effect_curve()):
        print(f"  {year}  {effect:7.2f}")

    # -- Figure 1 of Arkhangelsky et al. (2021)
    fig, _ = synthdid_plot(
        estimates,
        facet_vertical=False,
        control_name="control",
        treated_name="california",
        lambda_comparable=True,
        se_method="none",
        line_width=0.9,
        effect_curvature=-0.4,
        trajectory_alpha=0.7,
        effect_alpha=0.7,
        diagram_alpha=1.0,
        onset_alpha=0.7,
        figsize=(15, 5),
    )
    fig.savefig(FIGURES / "figure1.png", dpi=150, bbox_inches="tight")

    fig, _ = synthdid_plot(tau, se_method="placebo", se_replications=200)
    fig.savefig(FIGURES / "sdid.png", dpi=150, bbox_inches="tight")

    fig, _ = synthdid_plot(tau, overlay=1.0, se_method="none")
    fig.savefig(FIGURES / "overlay.png", dpi=150, bbox_inches="tight")

    fig, _ = synthdid_units_plot(tau, se_method="placebo", se_replications=200)
    fig.savefig(FIGURES / "units.png", dpi=150, bbox_inches="tight")

    fig, _ = tau.placebo_plot(se_method="none")
    fig.savefig(FIGURES / "placebo.png", dpi=150, bbox_inches="tight")

    plt.close("all")
    print(f"\nFigures written to {FIGURES}")


if __name__ == "__main__":
    main()
