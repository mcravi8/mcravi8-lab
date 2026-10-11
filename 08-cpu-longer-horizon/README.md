# Experiment 08: CPU longer-horizon and history test

The first stage tests numerical forecasts using the frozen Experiment 06 market snapshots. No JEV inference, GPU, Runpod operation, RL or downloads.

## V1 results: longer horizon and year view did not add skill

The protocol/source freeze was pushed in [c557f4d](https://github.com/mcravi8/mcravi8-lab/commit/c557f4d) before evaluating the real outcomes. **16,566 held-out observations across 335 weekly dates and 50 tickers** cover 2020-04-03 through 2026-08-28. Both horizons and all models use the same rows and purged annual folds. The last target is 2026-09-28; Experiment 05's future holdout is untouched.

| Horizon | Model | Accuracy | Brier |
| --- | --- | ---: | ---: |
| 5 sessions | Train-frequency prior | 40.61% | 0.658085 |
| 5 sessions | Current history | 36.50% | 0.721457 |
| 5 sessions | Current + year view | 35.56% | 0.759679 |
| 20 sessions | Train-frequency prior | 46.69% | 0.638167 |
| 20 sessions | Current history | 41.69% | 0.707507 |
| 20 sessions | Current + year view | 39.88% | 0.754241 |

Lower Brier is better. Twenty-session accuracy rises, but the training prior improves more; neither horizon establishes added forecasting skill. The 20-session current model trails the prior in all seven years. The year view improves on current history in only one year and trails the prior in all seven. Uniform and date-vector shuffled-label results are included in [the full report](results/evaluation_v1/README.md).

The two frozen decision comparisons failed. Positive Brier reduction means improvement:

| 20-session comparison | Brier reduction | 97.5% eight-date block-bootstrap CI |
| --- | ---: | --- |
| Current history vs prior | -0.069340 | [-0.108153, -0.037367] |
| Added year view vs current history | -0.046734 | [-0.085865, -0.012240] |

Intervals account for the two decision comparisons using Bonferroni marginal levels. The 4/12/52-date sensitivities also remain negative. Eleven CPU tests passed; all 33,132 horizon-specific targets, labels and dates were independently checked with pandas. Seven purged folds, prior counts, all 42 checkpoint reloads and saved metrics passed verification. See [validation.json](results/evaluation_v1/validation.json), [metrics.json](results/evaluation_v1/metrics.json) and [manifest.json](results/evaluation_v1/manifest.json).

**This CPU screen provides no evidence to justify further GPU or RL work under the frozen decisions.** It tests one fixed linear model on numerical features, not direct JEV inference at 20 sessions, named-pattern detectors or every possible model. No additional search was performed after observing the failures.

## Frozen CPU design and reproduction

Compare next-five-session and next-20-session Up/Flat/Down targets on identical naturally sampled actual week-end observations. At each horizon, one fixed logistic model sees current 5/20/100-session information; another adds a 252-session year view aggregated to weekly bars. Both get matching SPY context and the causal target sigma. This tests numerical input histories, not literal rendered panels or the predictive value of named detector patterns.

[protocol.json](protocol.json) fixes the source/data hashes, model settings, splits and decisions before outcome evaluation. Five preceding calendar years supply each annual fit; both horizons use the same conservative purge requiring every training target to precede the first test's 252-session information window. The eligible test history begins on/after 2019-04-01; annual evaluation is 2020-2026, with 2026 partial. All models/horizons use common observations. This is a larger weekly cohort than Experiment 07, not a matched replication of its 300 forecasts.

Flat is strictly below one-third of the sample std of 60 overlapping trailing returns of the forecast horizon. Five-session scale needs 65 closes; 20-session scale needs 80. Horizon-specific train-frequency priors and uniform probabilities are references. A date-vector shuffled-label fit is a negative control. Model/scaler fits use training only and checkpoint reloads must preserve probabilities within 1e-12. No parameter search or test-selected confidence threshold.

Two prespecified CPU decisions: current-history 20-session model versus its prior; year-view 20-session model versus current history. Each requires a positive 97.5% paired block-bootstrap interval and gains in at least five of seven years; the year view must also beat the prior. Bonferroni intervals account for the two decision comparisons. Eight retained weekly dates are primary, with 4/12/52-date sensitivities; all tickers at a date stay together. Raw accuracy or Brier across different horizons is not a paired skill comparison. A passed gate motivates independent confirmation, not an automatic GPU run.

Use the pinned Python 3.11 Experiment 06 environment from the repository root:

```sh
06-jev-patterns/.venv/bin/python -m unittest discover -s 08-cpu-longer-horizon/tests -v
06-jev-patterns/.venv/bin/python 08-cpu-longer-horizon/src/study.py --verify-only
06-jev-patterns/.venv/bin/python 08-cpu-longer-horizon/src/study.py
```

Recorded results are under `results/evaluation_v1/`; input features and fitted models are ignored under `data/` and `.cache/`. Existing results cannot be overwritten. A fresh clone needs the checksum-verified ignored Experiment 06 parquet files.

The longest-history purge removes roughly a year of recent training labels in most folds. It is deliberately stricter than fitting on all labels available at forecast time and can make annual models stale. Historical years and the surviving ticker universe have already informed research. Results are exploratory, conditional on a fixed linear model, with overlapping targets/history and approximate bootstrap assumptions. No refit uncertainty, trading execution or net P&L is evaluated. Experiment 05's prospective holdout and frozen model remain untouched.
