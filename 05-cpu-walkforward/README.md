# Experiment 05: CPU walk-forward baselines

Can inexpensive numerical forecasts beat training-frequency priors over longer histories, and does adding known volatility/context help beyond relative chart geometry?

The CPU study is complete. The tested logistic models underperform the pooled prior on both targets. The scientific improvement is a better-controlled experiment, not better measured forecast accuracy.

## Frozen protocol

[`protocol.json`](protocol.json) was committed and pushed in **`0d2fae5` before the new model results were inspected**. It fixes the tickers, features, tasks, model settings, annual folds, controls and future test dates. Its SHA-256 is `93eb54750f504e2c598f8420aa9af3997d789226eef20cdd16e762fe1df9b6ee`.

All observations through 2026-10-09 are development data. Even newly downloaded earlier history is exploratory because the research questions were informed by earlier experiments. No historical interval is presented as independent confirmation.

## Data, targets and models

The same 12 hand-selected tickers as Experiment 03 use adjusted Yahoo OHLCV from 2000-01-01 through 2026-10-09, beginning at the IPO where later. No missing prices are filled. There are 10,870 eligible chart observations across 1,139 dates. Input weeks must contain exactly Monday-Friday; the next calendar week's last close is the target, including shortened target weeks. Volatility is the trailing 12-week sample return standard deviation known at the input Friday.

Two tasks are evaluated separately:

- `ternary_volatility`: Up/Flat/Down, with strict `abs(return) < sigma/3` defining Flat. Nonpositive or nonfinite sigma fails.
- `binary_direction`: positive versus negative return; exactly zero returns are excluded. There are 30 such exclusions in the full feature dataset and 12 in development test folds.

The 17 annual test folds cover 2010-2026 (2026 is partial). Each fit uses the preceding five calendar years and at least 104 training dates. Training targets must finish before the first test input week starts. All tickers on a date stay together. Standardization is fitted on training rows only; no model or hyperparameter is selected using the test folds.

`logistic_shape` has 25 features: each visible day's OHLC normalized to the week's price range, plus volume normalized to the week's maximum. These encode the same relative geometry available in the image, without ticker identity or absolute price scale. `logistic_context` adds eight known-at-Friday quantities: current/prior weekly return, 4/12-week momentum, rv20, sigma, relative weekly range and a 60-session volume ratio. This is an information ablation; it has more context than the old JEV images.

Both use a train-only StandardScaler and regularized logistic regression with C=1, max_iter=2000 and seed=17. No tuning was performed after observing outcomes. Reference models are pooled and ticker-specific training frequencies, uniform probabilities, always-Up, and fixed momentum/reversal rules. The latter three directional rules are deterministic and are especially harsh under probability losses. The shuffled-label control permutes whole training-date label vectors among dates with identical ticker coverage; it preserves same-date cross-ticker structure and class counts, not temporal dependence. It is diagnostic, not a formal randomized null test.

## Development results

Brier is the sum of squared option-probability errors; lower is better. BSS is relative to the pooled training prior refitted independently in each fold. Accuracy ties break in option order, so uniform argmax also selects Up. Four-week intervals below are for **prior Brier minus model Brier**; positive values would favor the model.

| Task | Model | Test n | Accuracy | Brier | BSS | Brier improvement 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ternary_volatility | pooled_train_prior | 7502 | 40.46% | 0.660360 | +0.00% | [+0.000, +0.000] |
| ternary_volatility | ticker_train_prior | 7502 | 39.51% | 0.663070 | -0.41% | [-0.004, -0.001] |
| ternary_volatility | logistic_shape | 7502 | 37.75% | 0.672570 | -1.85% | [-0.017, -0.008] |
| ternary_volatility | logistic_context | 7502 | 38.00% | 0.671732 | -1.72% | [-0.017, -0.006] |
| ternary_volatility | logistic_shuffled_labels | 7502 | 37.24% | 0.677287 | -2.56% | [-0.022, -0.012] |
| binary_direction | pooled_train_prior | 7490 | 55.46% | 0.495978 | +0.00% | [+0.000, +0.000] |
| binary_direction | ticker_train_prior | 7490 | 54.41% | 0.498347 | -0.48% | [-0.004, -0.001] |
| binary_direction | logistic_shape | 7490 | 52.78% | 0.506736 | -2.17% | [-0.015, -0.006] |
| binary_direction | logistic_context | 7490 | 53.24% | 0.508527 | -2.53% | [-0.018, -0.007] |
| binary_direction | logistic_shuffled_labels | 7490 | 51.64% | 0.513765 | -3.59% | [-0.023, -0.012] |

The prespecified ternary context model improves Brier in only 2 of 17 test years; the shape-only model does so in 1 of 17. The binary context model improves in 2 of 17 years. No predictive improvement over the pooled prior is established. Raw per-observation forecasts, all nine baselines, balanced accuracy, log loss, annual results and 1/2/4/8-week sensitivity are saved in `results/development_predictions.csv.gz` and `results/metrics.json`.

Intervals use paired circular blocks of consecutive retained forecast dates, keeping all tickers together, with 2,000 resamples. There are 718 dates in the ternary test folds, not 7,502 independent observations. Holiday gaps make blocks variable in calendar duration. Aggregation across 17 years assumes enough stability for the bootstrap approximation; annual results expose regime differences. These are descriptive development comparisons, without a multiplicity-adjusted discovery claim.

## Untouched future test

Future input weeks end from **2026-10-16 through 2027-10-08**. Its fixed primary comparison is `ternary_volatility / logistic_context` against the frozen pooled training prior. The primary metric is paired Brier reduction; success requires its prespecified 95% four-week block interval to be wholly above zero. Accuracy, balanced accuracy and log loss are secondary.

Before development evaluation, the primary model was fitted once to the last five available years: 2,520 rows across 210 dates, with last target 2026-10-09. Its scaler, coefficients, label order and comparator prior were frozen in local ignored `models/prospective_context.json`, then reload-verified. The committed `results/frozen_model_manifest.json` pins its hash. The code refuses to replace it with a different model. Do not refit or choose another primary model after reviewing development results.

Future observations are not yet available. Forecast collection and final future scoring have not started; the protocol and fixed model are prepared. Preserve the local model and raw data snapshots. A fresh clone excludes these assets, and a later Yahoo revision may prevent byte-identical reconstruction. `predict_frozen` applies the exported model without fitting; future evaluation must use the frozen hash and record forecasts before their next-week outcomes are available.

## Reproduce on CPU

From this experiment directory, with Python 3.11:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/data.py --as-of 2026-10-09
.venv/bin/python src/features.py
.venv/bin/python src/baselines.py
.venv/bin/python -m unittest discover -s tests -v
```

Run against the preserved local snapshot for exact reproduction. Data, feature, protocol, source and prediction hashes are recorded in `results/`. Dependencies are pinned. Tests check future-price mutations, price/volume invariance, holiday target rules, purging, future-holdout exclusion, grouped label shuffling, train-only scaling and exported-model probability identity. No GPU, Runpod or JEV request is used.

## Limits

The universe is hand chosen from current survivors, includes one ETF, and is not representative of all historical listings. Yahoo's adjusted prices are retrospective and revisable, not point-in-time data. Early folds have fewer listed tickers. Longer history adds regimes but does not remove selection bias. The limited fixed models do not settle whether other features or models could learn a signal. No trading strategy, execution prices, costs or P&L was tested. The reserved future year is the remaining test of the primary forecast claim.
