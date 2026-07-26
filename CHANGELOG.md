# Changelog

## 0.1.0

First release.

- Full port of the R package `synthdid` 0.0.9: `synthdid_estimate`, `sc_estimate`,
  `did_estimate`, the Frank-Wolfe solvers (with and without covariates), the bootstrap,
  jackknife and placebo variance estimators, `synthdid_placebo`, `synthdid_effect_curve`,
  `synthdid_controls` and `summary`.
- Point estimates match R to five decimals on `california_prop99`.
- Matplotlib versions of `synthdid_plot`, `synthdid_units_plot`, `synthdid_placebo_plot`
  and `synthdid_rmse_plot`.
- Staggered adoption via `staggered_synthdid_estimate`, following the Stata `sdid` package,
  with `never_treated` and `not_yet_treated` control pools.
- `panel_matrices` accepts covariate columns, an extension over the R version.
- Bundled datasets: `california_prop99`, `CPS`, `PENN`.
