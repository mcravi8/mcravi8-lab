# CPU longer-horizon results v1

Retrospective exploratory numerical forecasts, with a protocol/source freeze pushed before deriving these outcomes. No JEV inference or RL training. The recorded source freeze is [c557f4d](https://github.com/mcravi8/mcravi8-lab/commit/c557f4d).

| Horizon | Model | N | Accuracy | Brier | Brier skill vs prior |
| --- | --- | ---: | ---: | ---: | ---: |
| 5 sessions | train_prior | 16566 | 40.61% | 0.658085 | +0.00% |
| 5 sessions | uniform | 16566 | 40.61% | 0.666667 | -1.30% |
| 5 sessions | current_history | 16566 | 36.50% | 0.721457 | -9.63% |
| 5 sessions | multiscale_history | 16566 | 35.56% | 0.759679 | -15.44% |
| 5 sessions | shuffled_multiscale | 16566 | 36.38% | 0.776294 | -17.96% |

5-session labels: {'Up': 6728, 'Flat': 4492, 'Down': 5346}; 335 forecast-date clusters. Test-majority accuracy 40.61% is descriptive only.

| Horizon | Model | N | Accuracy | Brier | Brier skill vs prior |
| --- | --- | ---: | ---: | ---: | ---: |
| 20 sessions | train_prior | 16566 | 46.69% | 0.638167 | +0.00% |
| 20 sessions | uniform | 16566 | 46.69% | 0.666667 | -4.47% |
| 20 sessions | current_history | 16566 | 41.69% | 0.707507 | -10.87% |
| 20 sessions | multiscale_history | 16566 | 39.88% | 0.754241 | -18.19% |
| 20 sessions | shuffled_multiscale | 16566 | 40.10% | 0.749513 | -17.45% |

20-session labels: {'Up': 7735, 'Flat': 3672, 'Down': 5159}; 335 forecast-date clusters. Test-majority accuracy 46.69% is descriptive only.

Lower Brier is better. Compare models within a horizon; raw accuracy/Brier across different targets is not a matched forecast comparison.

| Paired comparison | Brier reduction | 95% eight-date CI | 97.5% decision CI | Positive years |
| --- | ---: | --- | --- | ---: |
| 5:current_history_vs_train_prior | -0.063372 | [-0.084569, -0.046029] | [-0.088073, -0.043763] | 0/7 |
| 5:multiscale_history_vs_current_history | -0.038222 | [-0.057477, -0.020534] | [-0.060138, -0.018749] | 0/7 |
| 5:multiscale_history_vs_train_prior | -0.101594 | [-0.129446, -0.077718] | [-0.133384, -0.074904] | 0/7 |
| 20:current_history_vs_train_prior | -0.069340 | [-0.101265, -0.040355] | [-0.108153, -0.037367] | 0/7 |
| 20:multiscale_history_vs_current_history | -0.046734 | [-0.079288, -0.016277] | [-0.085865, -0.012240] | 1/7 |
| 20:multiscale_history_vs_train_prior | -0.116074 | [-0.160495, -0.077052] | [-0.166172, -0.073938] | 0/7 |

Longer-horizon CPU gate: **failed**. Year-view CPU gate: **failed**.

The two decision comparisons use Bonferroni 97.5% marginal intervals for a 5% family across the two gates, and require at least five positive annual comparisons. The year-view gate also requires beating the 20-session prior. These are approximate bootstrap intervals, not guaranteed familywise coverage.

Circular moving blocks carry all tickers at each date together; eight retained weekly dates are primary, with 4/12/52-date sensitivities in metrics.json. Twenty-session outcomes overlap. Intervals condition on fitted predictions and assume approximate stationarity; no refit uncertainty is included.

All arms share eligible rows, five-year annual training periods and the longest-history purge. Training targets for both horizons precede the first test information window. The year view uses 252 daily sessions aggregated into weekly OHLCV, including partial boundary weeks only through the chart end. Current features include 5/20/100-session views. A common SPY context and the causal target sigma are supplied.

Limitations: the shared longest-history purge excludes roughly a year of recent labels in most folds, making fits older than an operational as-of fit; fixed linear model, hand-selected surviving universe, adjusted close-to-close targets, adaptively reused historical years, overlapping histories, partial 2026. This is not a direct replication of the 300-chart JEV pilot, a pattern-detector ablation, or evidence about every possible model. No costs, executable entry, risk-adjusted P&L, GPU, HTTP inference or market-data download. Experiment 05 future holdout/model remain unchanged.
