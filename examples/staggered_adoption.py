"""Staggered adoption: cohorts that start treatment in different years.

Simulates a panel where three groups adopt a policy in 2006, 2009 and 2012 with
a true effect of 2.0, then recovers it.

Run with::

    python examples/staggered_adoption.py
"""

import numpy as np
import pandas as pd

from synthdid import staggered_synthdid_estimate

TRUE_EFFECT = 2.0


def simulate(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_control, T = 40, 20
    cohorts = {6: 5, 9: 6, 12: 4}  # adoption period -> number of units

    n = n_control + sum(cohorts.values())
    unit_effect = rng.normal(scale=4.0, size=n)
    trend = np.linspace(0, 8, T)
    factor = rng.normal(size=(n, 2)) @ rng.normal(size=(2, T))
    Y = unit_effect[:, None] + trend[None, :] + factor + rng.normal(scale=0.3, size=(n, T))

    adoption = np.full(n, -1)
    row = n_control
    for start, size in cohorts.items():
        adoption[row : row + size] = start
        Y[row : row + size, start:] += TRUE_EFFECT
        row += size

    records = [
        (f"unit{i:03d}", 2000 + t, Y[i, t], int(adoption[i] >= 0 and t >= adoption[i]))
        for i in range(n)
        for t in range(T)
    ]
    return pd.DataFrame(records, columns=["unit", "year", "y", "treated"])


def main() -> None:
    panel = simulate()

    est = staggered_synthdid_estimate(
        panel, unit="unit", time="year", outcome="y", treatment="treated"
    )
    print(est)
    print(f"\ntrue effect {TRUE_EFFECT}, estimated ATT {est.att:.3f}")

    se = est.se(method="jackknife")
    low, high = est.ci(method="jackknife")
    print(f"jackknife se {se:.3f}   95% CI ({low:.3f}, {high:.3f})")

    # Not-yet-treated units can also serve as controls; each cohort's window is
    # then truncated at the next adoption date so those controls stay untreated.
    alt = staggered_synthdid_estimate(
        panel, unit="unit", time="year", outcome="y", treatment="treated",
        control_pool="not_yet_treated",
    )
    print(f"\nusing not-yet-treated controls: ATT {alt.att:.3f}")
    print(alt.by_cohort.to_string(index=False))


if __name__ == "__main__":
    main()
