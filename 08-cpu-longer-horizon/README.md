# Experiment 08: CPU longer-horizon and history test

The first stage tests numerical forecasts using the frozen Experiment 06 market snapshots. No JEV inference, GPU, Runpod operation, RL or downloads.

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

Results will be saved under `results/evaluation_v1/`; input features and fitted models are ignored under `data/` and `.cache/`. Existing results cannot be overwritten. A fresh clone needs the checksum-verified ignored Experiment 06 parquet files.

Historical years and the surviving ticker universe have already informed research. Results are exploratory, conditional on a fixed linear model, with overlapping targets/history and approximate bootstrap assumptions. No refit uncertainty, trading execution or net P&L is evaluated. Experiment 05's prospective holdout and frozen model remain untouched.
