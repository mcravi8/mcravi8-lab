# Experiment 03: zero-shot JEV-27B-VL on weekly charts, 12 tickers pooled

## Question
Does Experiment 02's null result (zero-shot JEV-27B-VL on unlabeled Monday to Friday charts, predicting next week Up/Flat/Down) hold across a broader pool of 12 tickers? Does a simple calibration fit on train weeks help? And, as an exploratory question only, does performance vary with volatility?

## Design
- Tickers, one pool with no predefined groups: TSLA, NVDA, PLTR, COIN, AAPL, MSFT, AMZN, META, KO, JNJ, PG, SPY. Adjusted daily OHLCV from yfinance, 2024-01-02 to 2026-10-09 (696 sessions each).
- Charts, labels, Flat band, prompt and placebo are identical to 02 (see `../02-jev-weekly-charts/README.md`). Charts are full Monday to Friday weeks with no axes or text. The label is next week's last close vs this Friday's close, with Flat when |r| < 1/3 of the std of the 12 trailing weekly returns.
- Fixed date split: train covers weeks ending 2024-04-05 to 2025-10-03 (64 charts per ticker). The week ending 2025-10-10 is dropped as a gap. Test covers 2025-10-17 to 2026-10-02 (41 per ticker). Totals: 768 train, 492 test, 1260 charts.
- Predictions: the 210 TSLA/NVDA charts were byte-identical to 02's (same SHA-256 and labels), so 02's predictions were reused. The other 1050 charts and their mirrored placebos were scored with `/v1/decide`, with 0 failures. `results/predictions.csv` holds every chart's labels, volatility features, raw, placebo and calibrated probabilities, and the source of each prediction.
- Metrics: as in 02. Hit rate is argmax. The honest base rate applies each ticker's train-majority class; the look-ahead base rate uses the test-majority class. The Brier reference is each ticker's train class frequencies, and BSS = 1 - Brier_model / Brier_ref. CIs use 2000 bootstrap resamples (seed 0): iid rows per ticker, and resampling whole weeks for pooled results so all tickers in a week move together. Success requires all three: the CI of hit minus test-majority base above zero, hit rate > placebo, and BSS > 0.
- Calibration: a multinomial logistic regression on log(p_up, p_flat, p_down), fit on pooled train rows only and applied to test.
- Volatility features per chart (`src/vol_features.py`), all known at that Friday's close:
  - rv20: annualized std of the 20 daily returns ending Friday.
  - rv20_pct: percentile of rv20 within the ticker's own history up to that date. The history is short early in the sample, since rv20 starts in 2024-01.
  - vix: ^VIX close that Friday.
  - spy_rv20: SPY's rv20.
  - Regimes: terciles of each feature, with cutoffs from train rows only (`results/vol_features.json`).

## Results: test period (primary)
| Ticker | Train vol | Hit | Base (train maj.) | Base (test maj.) | 95% CI hit - train base | 95% CI hit - test base | Placebo | Brier model / ref | BSS | Calibrated hit / BSS | All criteria |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TSLA | 68% | 0.439 | 0.366 | 0.366 | [-0.098, +0.244] | [-0.098, +0.244] | 0.415 | 0.816 / 0.665 | -0.226 | 0.341 / +0.028 | no |
| NVDA | 53% | 0.317 | 0.244 | 0.390 | [-0.146, +0.293] | [-0.244, +0.098] | 0.439 | 0.983 / 0.671 | -0.465 | 0.415 / -0.005 | no |
| PLTR | 66% | 0.268 | 0.341 | 0.341 | [-0.268, +0.098] | [-0.268, +0.098] | 0.366 | 0.982 / 0.683 | -0.439 | 0.268 / +0.007 | no |
| COIN | 81% | 0.293 | 0.366 | 0.366 | [-0.195, +0.049] | [-0.195, +0.049] | 0.341 | 1.026 / 0.666 | -0.540 | 0.317 / -0.011 | no |
| AAPL | 30% | 0.439 | 0.463 | 0.463 | [-0.171, +0.146] | [-0.171, +0.146] | 0.366 | 0.774 / 0.646 | -0.198 | 0.415 / +0.012 | no |
| MSFT | 23% | 0.317 | 0.317 | 0.390 | [-0.171, +0.171] | [-0.268, +0.122] | 0.488 | 0.926 / 0.671 | -0.380 | 0.244 / -0.004 | no |
| AMZN | 32% | 0.390 | 0.366 | 0.366 | [-0.244, +0.293] | [-0.244, +0.293] | 0.341 | 0.919 / 0.673 | -0.365 | 0.415 / +0.031 | no |
| META | 35% | 0.439 | 0.341 | 0.439 | [-0.098, +0.293] | [-0.195, +0.195] | 0.317 | 0.696 / 0.649 | -0.073 | 0.366 / -0.014 | no |
| KO | 16% | 0.366 | 0.463 | 0.463 | [-0.268, +0.074] | [-0.268, +0.074] | 0.317 | 0.882 / 0.653 | -0.349 | 0.415 / +0.015 | no |
| JNJ | 19% | 0.366 | 0.415 | 0.415 | [-0.220, +0.122] | [-0.220, +0.122] | 0.415 | 0.898 / 0.653 | -0.375 | 0.390 / -0.006 | no |
| PG | 18% | 0.415 | 0.341 | 0.341 | [-0.073, +0.220] | [-0.073, +0.220] | 0.341 | 0.847 / 0.699 | -0.212 | 0.317 / +0.034 | no |
| SPY | 18% | 0.415 | 0.415 | 0.415 | [-0.171, +0.171] | [-0.171, +0.171] | 0.341 | 0.786 / 0.663 | -0.187 | 0.390 / -0.001 | no |
| Pooled |  | 0.372 | 0.370 | 0.396 | [-0.057, +0.061] | [-0.077, +0.031] | 0.374 | 0.878 / 0.666 | -0.318 | 0.358 / +0.007 | no |


## Results: full sample (secondary)
| Ticker | Hit | Base (train maj.) | Base (full maj.) | 95% CI hit - full base | Placebo | BSS |
|---|---|---|---|---|---|---|
| TSLA | 0.400 | 0.371 | 0.371 | [-0.076, +0.133] | 0.352 | -0.316 |
| NVDA | 0.333 | 0.305 | 0.352 | [-0.133, +0.095] | 0.333 | -0.383 |
| PLTR | 0.305 | 0.400 | 0.400 | [-0.200, +0.000] | 0.381 | -0.464 |
| COIN | 0.295 | 0.390 | 0.390 | [-0.210, +0.010] | 0.314 | -0.411 |
| AAPL | 0.362 | 0.429 | 0.429 | [-0.171, +0.038] | 0.410 | -0.363 |
| MSFT | 0.371 | 0.343 | 0.343 | [-0.076, +0.133] | 0.352 | -0.301 |
| AMZN | 0.371 | 0.400 | 0.400 | [-0.190, +0.133] | 0.267 | -0.432 |
| META | 0.457 | 0.362 | 0.400 | [-0.067, +0.171] | 0.333 | -0.174 |
| KO | 0.381 | 0.410 | 0.410 | [-0.133, +0.076] | 0.314 | -0.314 |
| JNJ | 0.400 | 0.438 | 0.438 | [-0.143, +0.067] | 0.333 | -0.276 |
| PG | 0.448 | 0.429 | 0.429 | [-0.086, +0.133] | 0.324 | -0.213 |
| SPY | 0.495 | 0.429 | 0.429 | [-0.029, +0.171] | 0.352 | -0.118 |
| Pooled | 0.385 | 0.392 | 0.399 | [-0.053, +0.025] | 0.339 | -0.314 |


The model almost never predicts Flat: 2 of 492 test charts, despite Flat being about a third of the labels. That is the main reason its raw BSS is strongly negative.

**Calibration effect (pooled test):**
- Raw: hit rate 0.372, BSS -0.318.
- Calibrated: hit rate 0.358, BSS +0.007 [-0.014, +0.027].
- The fitted coefficients are small (|coef| <= 0.16), so the calibrated forecast is close to the train class frequencies. It removes JEV's overconfidence but adds no measurable skill.

**Success criteria:** met by no ticker and not by the pooled set. Every CI includes zero and every raw BSS is negative.

## Exploratory analyses (not pre-registered; treat as hypothesis-generating)
These include many comparisons: 2 cross-ticker correlations, 18 tercile intervals and 4 logit coefficients, with no correction for multiple comparisons. About 1 in 20 intervals would exclude zero by chance alone.

**Cross-ticker** (n = 12): Spearman correlation between train-period annualized vol and test results, with 95% CIs from resampling tickers.
- Vs test edge (hit rate minus train base): rho = +0.19 [-0.58, +0.81].
- Vs raw test BSS: rho = -0.44 [-0.84, +0.21].
- Neither is distinguishable from zero.

**By volatility tercile** (test period, pooled, week-block bootstrap; see `results/vol_regimes.png`):
| Measure | Tercile | n (weeks) | Hit - train base [95% CI] | BSS [95% CI] |
|---|---|---|---|---|
| rv20 | low | 139 (41) | +0.000 [-0.092, +0.092] | -0.313 [-0.471, -0.156] |
| rv20 | mid | 190 (41) | +0.021 [-0.102, +0.129] | -0.214 [-0.397, -0.041] |
| rv20 | high | 163 (41) | -0.018 [-0.115, +0.069] | -0.441 [-0.589, -0.295] |
| rv20_pct | low | 127 (38) | +0.087 [-0.021, +0.197] | -0.247 [-0.420, -0.070] |
| rv20_pct | mid | 229 (41) | -0.066 [-0.146, +0.026] | -0.388 [-0.499, -0.265] |
| rv20_pct | high | 136 (38) | +0.037 [-0.047, +0.125] | -0.268 [-0.442, -0.071] |
| vix | low | 120 (10) | -0.058 [-0.158, +0.042] | -0.544 [-0.683, -0.424] |
| vix | mid | 156 (13) | +0.006 [-0.077, +0.096] | -0.278 [-0.493, -0.090] |
| vix | high | 216 (18) | +0.032 [-0.065, +0.134] | -0.218 [-0.356, -0.082] |


No tercile shows an edge whose CI excludes zero. Raw BSS is negative in every tercile.

**Logit of correct (0/1)** on the standardized vol features plus ticker fixed effects (test rows, n = 492, SEs clustered by week):
| Feature (standardized) | Coef | 95% CI | p |
|---|---|---|---|
| rv20 | -0.193 | [-1.024, +0.638] | 0.65 |
| rv20_pct | +0.013 | [-0.415, +0.440] | 0.95 |
| vix | +0.236 | [-0.075, +0.547] | 0.14 |
| spy_rv20 | +0.214 | [-0.724, +1.152] | 0.65 |

No coefficient is distinguishable from zero. rv20, spy_rv20 and vix are correlated, so the individual coefficients are unstable.

## Figures
- `results/ticker_edge.png`: per-ticker test edge with CIs, sorted by train-period vol, plus raw vs calibrated BSS.
- `results/vol_regimes.png`: edge by tercile for rv20, rv20_pct and VIX.

## Limitations
- 41 test weeks per ticker, so per-ticker CIs are about +/-0.17 wide. Pooled tickers are strongly correlated within a week: 492 rows but only 41 independent weeks.
- Zero-shot with one prompt. The Flat band is defined in terms of volatility, and the model has no notion of it.
- The test period is one year in one market regime. The VIX terciles contain only 10 to 18 weeks each.
- The ticker list was chosen by hand. Volatility rankings are used only descriptively and only from train data.
- Calibration has 3 x 4 parameters, fit on 768 train rows. It was not tuned further.

## Reproduce
```sh
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/data.py && .venv/bin/python src/render.py && .venv/bin/python src/vol_features.py
# serve as in 01/02 (scripts/remote_serve.sh), then score charts not reused from 02:
JEV_BASE_URL=https://<POD_ID>-8000.proxy.runpod.net .venv/bin/python src/score.py --api decide --index charts/index_new.csv
JEV_BASE_URL=... .venv/bin/python src/score.py --api decide --placebo --index charts/index_new.csv
.venv/bin/python src/evaluate.py
```
`charts/index_new.csv` is `charts/index.csv` without TSLA/NVDA. Run 2026-10-10 on one NVIDIA H100 80GB HBM3 pod (Runpod Secure Cloud, US-MO-1, $3.99/hr; A100 80GB was unavailable) for 11 min 35 s, about $0.77.
