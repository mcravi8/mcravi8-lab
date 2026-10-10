# Experiment 02: JEV on weekly TSLA and NVDA charts

Can zero-shot JEV forecast next-week Up/Flat/Down from one week of chart geometry? The historical run provides no demonstrated advantage over training-selected baselines.

## Design

Charts show exactly one full Monday-Friday week. Holiday-shortened input weeks are excluded. The target is the next calendar week's last available close divided by the chart Friday close, minus one, including shortened target weeks. Flat means `abs(return) < (1/3) * sigma`, where sigma is the sample standard deviation of the last 12 completed weekly returns ending at the chart Friday. Equality is directional. Nonfinite or nonpositive sigma now raises an error; existing real labels are unchanged.

The split is by date: 64 training chart weeks per ticker from 2024-04-05 through 2025-10-03; the week ending 2025-10-10 is omitted; 41 test weeks run from 2025-10-17 through 2026-10-02. Input and target boundaries were checked for overlap. Historical JEV used an unlabeled, independently scaled 640 x 480 image with the question: "This is a 5-day daily stock chart covering one trading week. Over the next week, will the price go Up, stay Flat, or go Down?" That prompt did not define Flat or supply volatility.

TSLA and NVDA each have 696 adjusted daily sessions from 2024-01-02 through 2026-10-09. Train/test labels (Up, Flat, Down) are TSLA 24/20/20 and 15/14/12; NVDA 20/22/22 and 16/10/15. Training-majority ties break Up, Flat, Down, selecting Flat for NVDA.

## Corrected historical results

CIs use paired circular moving blocks of four consecutive retained forecast weeks, carrying all tickers in a date together. JSON also reports block lengths 1, 2 and 8, paired Brier/accuracy differences, and date-level sign-flip tests. Holiday gaps mean four retained dates can span more than four calendar weeks. Blocks address dependence approximately; stationarity is an assumption, and sign-flip tests additionally assume independent symmetric date differences. Exact enumeration does not remove those assumptions.

The primary references are selected or fitted on training labels. Test-majority rates are descriptive look-ahead benchmarks only, with no CI or success claim based on them. All historical results are exploratory because the test period informed successive experiments. Intervals remain unadjusted and exploratory. Selected families have Holm-adjusted p-values: four volatility coefficients in 03, and six model pairs across two metrics in 04. These limited adjustments do not correct the broader adaptive research process.

| Ticker | Test n | Hit | Train-selected base | Hit minus base 95% CI | Mirrored hit | Brier | BSS |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| TSLA | 41 | 0.439 | 0.366 | [-0.122, +0.293] | 0.415 | 0.816 | -0.226 |
| NVDA | 41 | 0.317 | 0.244 | [-0.146, +0.317] | 0.439 | 0.983 | -0.465 |

Neither ticker passes the revised evidence checks. JEV never chose Flat by argmax. Undefined Flat in the prompt and missing absolute scale are plausible task mismatches, not a proven explanation for the result. The full sample is secondary and contains training rows; its detailed metrics remain in `results/metrics.json`.

## Reproduce

CPU evaluation reads existing local charts and historical prediction artifacts:

```sh
python src/data.py --as-of 2026-10-09
python src/render.py
python src/evaluate.py
```

Install `requirements.txt` first. To collect new probabilities later, follow Experiment 01's serving instructions and use a fresh directory, for example `python src/score.py --task ternary-volatility --results-dir results/revised_ternary`. Run matching `--blank` and `--placebo` controls and the separate binary task. These revised real-model runs have not been executed. Historical inference ran on an H100 in US-MO-1; the current correction did not start a pod.

The sample is small, the tickers were chosen by hand, adjusted Yahoo history can be revised, and the reused test year is not independent confirmation. No strategy, transaction costs or trading returns were tested.
