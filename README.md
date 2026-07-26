# synthdid-py

**Synthetic Difference-in-Differences for Python** — a complete port of the reference R package
[`synthdid`](https://github.com/synth-inference/synthdid) by Arkhangelsky, Athey, Hirshberg,
Imbens and Wager.

Everything the R package does, this does: the estimator, the Frank-Wolfe solvers, covariate
adjustment, all three variance estimators, the placebo diagnostic, and the full set of plots.
Plus staggered adoption, which the R package does not support.

Point estimates agree with R to the fifth decimal on the canonical example.

日本語版の README は [README.ja.md](README.ja.md) にあります。

---

## Install

```bash
pip install synthdid-py
```

Plots need matplotlib:

```bash
pip install "synthdid-py[plot]"
```

## Quickstart

```python
from synthdid import load_california_prop99, panel_matrices, synthdid_estimate

panel = panel_matrices(load_california_prop99())   # long panel -> Y, N0, T0, W
tau = synthdid_estimate(panel.Y, panel.N0, panel.T0,
                        unit_names=panel.units, time_labels=panel.time)

print(tau)
# synthdid_estimate: -15.6038. Effective N0/N0 = 16.4/38. Effective T0/T0 = 2.8/19. N1,T1 = 1,12.

print(tau.se(method="placebo"))    # ~9.8  (California is the only treated unit)
print(tau.ci(method="placebo"))    # ~(-34.8, 3.6)
```

`tau` **is a float** — compare it, add it, put it in a DataFrame — that also carries
`.weights`, `.setup` and `.opts`.

```python
tau.summary()          # estimate, se, effective sample sizes, leading weights
tau.controls()         # the control units that carry the weight, sorted
tau.effect_curve()     # the per-period effects that average to the estimate
tau.placebo()          # refit on pre-treatment data only
tau.plot()             # the trajectory + 2x2 diagram figure
```

## Verified against the R implementation

Run on `california_prop99` (39 states, 1970-2000, California treated from 1989):

| estimator                   | this package | R `synthdid` 0.0.9 |
|-----------------------------|-------------:|-------------------:|
| `synthdid_estimate`         |   −15.60383  |          −15.60383 |
| `sc_estimate`               |   −19.61966  |          −19.61966 |
| `did_estimate`              |   −27.34911  |          −27.34911 |

The test suite also checks, on synthetic panels, every invariance the R package tests for
(unit and time fixed effects, rescaling, block shifts), and validates the Frank-Wolfe solver
against an independent projected-gradient solver with a duality-gap optimality certificate.

```bash
pytest        # 108 tests
```

## The three estimators

```python
from synthdid import synthdid_estimate, sc_estimate, did_estimate

sdid = synthdid_estimate(Y, N0, T0)   # unit weights AND time weights
sc   = sc_estimate(Y, N0, T0)         # unit weights only  (synthetic control)
did  = did_estimate(Y, N0, T0)        # uniform weights    (difference-in-differences)
```

`Y` is an `N x T` array whose first `N0` rows are the control units and whose first `T0`
columns are the pre-treatment periods. A DataFrame works too, and its index and columns become
the unit and period labels.

## Standard errors

| method | algorithm in the paper | when to use |
|---|---|---|
| `"bootstrap"` | Algorithm 2 | the default; resamples units |
| `"jackknife"` | Algorithm 3 | fast; **not** recommended for synthetic control |
| `"placebo"`   | Algorithm 4 | the only option with a single treated unit |

```python
tau.se(method="bootstrap", replications=200, random_state=0)
tau.vcov(method="jackknife")            # 1x1 matrix, like R's vcov
tau.ci(level=0.95, method="placebo")
```

With one treated unit the bootstrap and jackknife are undefined and return `nan` — same as R.
Pass `random_state` for reproducible resampling.

## Covariates

Pass a time-varying `N x T x C` array. Weights and coefficients are solved jointly, as in
Arkhangelsky et al.'s "optimized" adjustment:

```python
panel = panel_matrices(df, unit="country", time="year",
                       outcome="log_gdp", treatment="dem",
                       covariates=["educ"])
tau = synthdid_estimate(panel.Y, panel.N0, panel.T0, X=panel.X)
tau.weights.beta        # the covariate coefficients
```

## Staggered adoption

Units that start treatment at different times — the case the R package rejects. Cohorts are
estimated separately and combined with weights proportional to treated unit-periods, following
the Stata `sdid` package:

```python
from synthdid import staggered_synthdid_estimate

est = staggered_synthdid_estimate(df, unit="state", time="year",
                                  outcome="y", treatment="d")
print(est.att)                                  # aggregate ATT
print(est.by_cohort)                            # one row per adoption date
print(est.se(method="jackknife"))
est.estimates[2005].plot()                      # drill into one cohort
```

`control_pool="never_treated"` (default) uses only never-treated units, matching Stata.
`control_pool="not_yet_treated"` also admits units that adopt later, truncating each cohort's
post-period at the next adoption date so those controls stay clean.

With a single adoption date this reduces exactly to `synthdid_estimate`.

## Plots

```python
tau.plot()                      # trajectories + parallelogram + effect arrow + lambda ribbon
tau.units_plot()                # each control's own diff-in-diff, sized by its weight
tau.placebo_plot()              # the estimate beside a pre-treatment placebo
tau.rmse_plot()                 # solver convergence diagnostic

synthdid_plot({"DiD": did, "SC": sc, "SDID": sdid})                 # one panel each
synthdid_plot({"DiD": did, "SC": sc, "SDID": sdid}, facet=["a"]*3)  # overlaid
```

Every plotting function returns `(fig, ax)`, so styling is ordinary matplotlib.
`examples/quickstart.py` reproduces Figure 1 of the paper.

## API

| | |
|---|---|
| **Estimation** | `synthdid_estimate` · `sc_estimate` · `did_estimate` · `staggered_synthdid_estimate` |
| **Inference** | `vcov` · `synthdid_se` · `bootstrap_se` · `jackknife_se` · `placebo_se` |
| **Diagnostics** | `summary` · `synthdid_controls` · `synthdid_placebo` · `synthdid_effect_curve` |
| **Plots** | `synthdid_plot` · `synthdid_units_plot` · `synthdid_placebo_plot` · `synthdid_rmse_plot` |
| **Data** | `panel_matrices` · `PanelData` · `random_low_rank` · `load_california_prop99` · `load_cps` · `load_penn` |
| **Solver** | `sc_weight_fw` · `sc_weight_fw_covariates` · `fw_step` · `sparsify_function` · `contract3` |

## Notes for people coming from R

* `synthdid_estimate(Y, N0, T0)` returns a float subclass, not a scalar with attributes.
  `attr(est, "weights")` becomes `est.weights`; `attr(est, "setup")` becomes `est.setup`.
* `weights$lambda` is `weights.lambda_` (`lambda` is a Python keyword). `weights["lambda"]`
  also works, as do the aliases `weights.time` and `weights.unit`.
* `vcov()` returns a 1×1 array so it stays a drop-in match; `.se()` gives the scalar directly.
* `print(tau.hat)` in R runs a jackknife. Here `repr` stays cheap and computes no standard
  error; use `tau.format()` for the R-style one-liner or `tau.summary()` for the full report.
* Column selectors in `panel_matrices` are 1-based integers *or* names, as in R.
* Resampling takes an explicit `random_state` rather than reading a global seed.

## Citing

If you use this package, cite the method:

> Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021).
> Synthetic Difference-in-Differences. *American Economic Review*, 111(12), 4088–4118.

## License

BSD 3-Clause. This is a derivative work of the R package `synthdid`
(© 2019 Stanford University), which is dual-licensed GPL (≥2) | BSD 3-Clause; this port takes
the BSD option. See [NOTICE](NOTICE) for full attribution, including the Stata `sdid` package
whose staggered-adoption design this follows, and the sources of the bundled datasets.
