# synthdid-py

[![PyPI](https://img.shields.io/pypi/v/synthdid-py.svg)](https://pypi.org/project/synthdid-py/)
[![Tests](https://github.com/tomtomtom1007/synthdid-py/actions/workflows/ci.yml/badge.svg)](https://github.com/tomtomtom1007/synthdid-py/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/synthdid-py.svg)](https://pypi.org/project/synthdid-py/)
[![License](https://img.shields.io/pypi/l/synthdid-py.svg)](https://github.com/tomtomtom1007/synthdid-py/blob/main/LICENSE)

**Synthetic Difference-in-Differences (合成群間差分の差)** の Python 実装。
Arkhangelsky, Athey, Hirshberg, Imbens, Wager による参照実装
[R パッケージ `synthdid`](https://github.com/synth-inference/synthdid) の完全移植です。

R 版にある機能はすべて入っています — 推定量、Frank-Wolfe ソルバー、共変量調整、
3 種類の分散推定、プラセボ診断、作図一式。加えて、R 版が対応していない
**staggered adoption（処置開始時期がずれるケース）** にも対応しています。

代表例における点推定値は R 版と小数第 5 位まで一致します。

---

## インストール

```bash
pip install synthdid-py
```

作図には matplotlib が必要です：

```bash
pip install "synthdid-py[plot]"
```

## クイックスタート

```python
from synthdid import load_california_prop99, panel_matrices, synthdid_estimate

panel = panel_matrices(load_california_prop99())   # long 形式 -> Y, N0, T0, W
tau = synthdid_estimate(panel.Y, panel.N0, panel.T0,
                        unit_names=panel.units, time_labels=panel.time)

print(tau)
# synthdid_estimate: -15.6038. Effective N0/N0 = 16.4/38. Effective T0/T0 = 2.8/19. N1,T1 = 1,12.

print(tau.se(method="placebo"))    # 約 9.8（処置群はカリフォルニア 1 州のみ）
print(tau.ci(method="placebo"))    # 約 (-34.8, 3.6)
```

`tau` は **float のサブクラス** です。そのまま比較・演算・DataFrame への格納ができ、
同時に `.weights` / `.setup` / `.opts` を持ちます。

```python
tau.summary()          # 推定値・標準誤差・実効サンプルサイズ・主要ウェイト
tau.controls()         # ウェイトを持つコントロール群（降順）
tau.effect_curve()     # 平均すると推定値になる期別効果
tau.placebo()          # 処置前データのみでの再推定
tau.plot()             # トラジェクトリ + 2x2 ダイアグラム図
```

## R 実装との一致

`california_prop99`（39 州・1970-2000 年・カリフォルニアが 1989 年から処置）での結果：

| 推定量                       | 本パッケージ | R `synthdid` 0.0.9 |
|-----------------------------|-------------:|-------------------:|
| `synthdid_estimate`         |   −15.60383  |          −15.60383 |
| `sc_estimate`               |   −19.61966  |          −19.61966 |
| `did_estimate`              |   −27.34911  |          −27.34911 |

テストスイートでは、R 版が検証している不変性（ユニット固定効果・時間固定効果・
スケール変換・ブロックシフト）をすべて人工パネル上で確認し、さらに Frank-Wolfe
ソルバーを独立な射影勾配法ソルバー（双対ギャップによる最適性証明つき）と突き合わせています。

```bash
pytest        # 108 tests
```

## 3 つの推定量

```python
from synthdid import synthdid_estimate, sc_estimate, did_estimate

sdid = synthdid_estimate(Y, N0, T0)   # ユニットウェイト + 時間ウェイト
sc   = sc_estimate(Y, N0, T0)         # ユニットウェイトのみ（合成コントロール）
did  = did_estimate(Y, N0, T0)        # 一様ウェイト（通常の DiD）
```

`Y` は `N x T` 配列で、先頭 `N0` 行がコントロール群、先頭 `T0` 列が処置前期間です。
DataFrame を渡すと、index と columns がそのままユニット名・期ラベルになります。

## 標準誤差

| method | 論文中のアルゴリズム | 使いどころ |
|---|---|---|
| `"bootstrap"` | Algorithm 2 | 既定。ユニット単位のリサンプリング |
| `"jackknife"` | Algorithm 3 | 高速。ただし合成コントロールには**非推奨** |
| `"placebo"`   | Algorithm 4 | 処置ユニットが 1 つのときの唯一の選択肢 |

```python
tau.se(method="bootstrap", replications=200, random_state=0)
tau.vcov(method="jackknife")            # R の vcov と同じく 1x1 行列
tau.ci(level=0.95, method="placebo")
```

処置ユニットが 1 つの場合、bootstrap と jackknife は定義されず `nan` を返します（R 版と同じ挙動）。
再現性が必要なときは `random_state` を渡してください。

## 共変量

時間変動する `N x T x C` 配列を渡します。ウェイトと係数は同時に解かれます
（Arkhangelsky らの "optimized" 方式）。

```python
panel = panel_matrices(df, unit="country", time="year",
                       outcome="log_gdp", treatment="dem",
                       covariates=["educ"])
tau = synthdid_estimate(panel.Y, panel.N0, panel.T0, X=panel.X)
tau.weights.beta        # 共変量の係数
```

## Staggered adoption（処置開始時期がずれる場合）

R 版がエラーにするケースです。採用時期ごとのコホートを個別に推定し、
処置ユニット×処置後期間数に比例したウェイトで統合します（Stata の `sdid` に準拠）。

```python
from synthdid import staggered_synthdid_estimate

est = staggered_synthdid_estimate(df, unit="state", time="year",
                                  outcome="y", treatment="d")
print(est.att)                                  # 集計 ATT
print(est.by_cohort)                            # 採用時期ごとの内訳
print(est.se(method="jackknife"))
est.estimates[2005].plot()                      # 個別コホートを掘り下げる
```

`control_pool="never_treated"`（既定）は一度も処置されないユニットのみを対照に使い、Stata と一致します。
`control_pool="not_yet_treated"` は後から処置されるユニットも対照に含め、そのぶん各コホートの
処置後期間を次の採用時期の直前で打ち切ります。

採用時期が 1 つだけの場合、結果は `synthdid_estimate` と完全に一致します。

## 作図

```python
tau.plot()                      # トラジェクトリ + 平行四辺形 + 効果の矢印 + λ リボン
tau.units_plot()                # コントロール別 DiD（点の大きさがウェイト）
tau.placebo_plot()              # 推定値とプラセボ推定の並置
tau.rmse_plot()                 # ソルバー収束の診断

synthdid_plot({"DiD": did, "SC": sc, "SDID": sdid})                 # パネル分割
synthdid_plot({"DiD": did, "SC": sc, "SDID": sdid}, facet=["a"]*3)  # 重ね描き
```

作図関数はすべて `(fig, ax)` を返すので、スタイル調整は通常の matplotlib と同じです。
`examples/quickstart.py` は論文の Figure 1 を再現します。

## API 一覧

| | |
|---|---|
| **推定** | `synthdid_estimate` · `sc_estimate` · `did_estimate` · `staggered_synthdid_estimate` |
| **推論** | `vcov` · `synthdid_se` · `bootstrap_se` · `jackknife_se` · `placebo_se` |
| **診断** | `summary` · `synthdid_controls` · `synthdid_placebo` · `synthdid_effect_curve` |
| **作図** | `synthdid_plot` · `synthdid_units_plot` · `synthdid_placebo_plot` · `synthdid_rmse_plot` |
| **データ** | `panel_matrices` · `PanelData` · `random_low_rank` · `load_california_prop99` · `load_cps` · `load_penn` |
| **ソルバー** | `sc_weight_fw` · `sc_weight_fw_covariates` · `fw_step` · `sparsify_function` · `contract3` |

## R 版からの移行メモ

* `synthdid_estimate(Y, N0, T0)` は属性つきスカラーではなく float サブクラスを返します。
  `attr(est, "weights")` → `est.weights`、`attr(est, "setup")` → `est.setup`。
* `weights$lambda` は `weights.lambda_`（`lambda` は Python の予約語のため）。
  `weights["lambda"]` でもアクセスでき、`weights.time` / `weights.unit` という別名もあります。
* `vcov()` は R と揃えて 1×1 配列を返します。スカラーが欲しいときは `.se()` を使ってください。
* R の `print(tau.hat)` は内部で jackknife を走らせます。本パッケージの `repr` は
  標準誤差を計算せず軽量に保っています。R 形式の 1 行表示は `tau.format()`、
  詳細は `tau.summary()` です。
* `panel_matrices` の列指定は R と同じく **1 始まりの整数**または列名です。
* リサンプリングはグローバルシードではなく `random_state` を明示的に受け取ります。

## 引用

本パッケージを使う場合は手法そのものを引用してください：

> Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021).
> Synthetic Difference-in-Differences. *American Economic Review*, 111(12), 4088–4118.

## ライセンス

BSD 3-Clause。本パッケージは R パッケージ `synthdid`（© 2019 Stanford University、
GPL (≥2) | BSD 3-Clause のデュアルライセンス）の派生物であり、BSD 側を選択しています。
帰属の詳細（staggered adoption の設計を踏襲した Stata `sdid`、および同梱データの出典を含む）は
[NOTICE](NOTICE) を参照してください。
