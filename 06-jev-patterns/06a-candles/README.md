# 06a: Candlestick recognition dataset

Day 1 is CPU-only. The dataset contains **1,200 unlabelled 640 x 480 PNGs**, with 150 examples of each of eight classes. Every class has 75 training and 75 test examples. No JEV inference or GPU training has run for this dataset.

The task is recognition of the pattern ending at the right edge of a 20-candle chart, with volume below. It does not predict subsequent returns. All raw inputs, visible numeric bars, and full chart collections are ignored by Git; the selected index, provenance, statistics, nine samples and audit sheets are committed.

## Labels and frozen rules

[rules.json](rules.json) was written and hashed before scanning or sampling the downloaded data. It records the precise thresholds, trend definition, embargo, deterministic sampler and upstream source hashes. The eight options, in scoring order, are in [options.json](options.json).

TA-Lib **0.8.1** supplies `CDLDOJI`, `CDLHAMMER`, `CDLSHOOTINGSTAR`, the signed `CDLENGULFING`, `CDLMORNINGSTAR` and `CDLEVENINGSTAR`. A separately implemented pandas detector uses rolling OHLC arithmetic and no TA-Lib calls. Definitions were checked against the [matching official TA-Lib release](https://github.com/TA-Lib/ta-lib/tree/v0.8.1/src/ta_func) and its [default candle settings](https://github.com/TA-Lib/ta-lib/blob/v0.8.1/src/ta_common/ta_global.c).

Some defaults differ from informal charting descriptions. A Hammer's lower wick must exceed its body, rather than twice the body. Its body must be near the preceding low. Shooting Star requires a strict upward body gap. Stars require a gapped short middle body and 30% penetration of the first body by a third body larger than its prior ten-body average. Engulfing accepts a shared endpoint when the other endpoint extends strictly.

Hammer and Morning Star require a negative five-session return before the pattern starts; Shooting Star and Evening Star require a positive one. For a pattern spanning `s` candles and ending at index `t`, this is `Close[t-s] / Close[t-s-5] - 1`. Zero trend fails. Doji and Engulfing have no added trend filter. The [context prompt](../shared/context/candles.txt) describes these rules.

We retain a window only if both detectors return exactly the same complete set of pattern flags and that set contains exactly one pattern, or no pattern. **None means none of these seven patterns**, not absence of every conceivable candlestick formation. Multiple matches are excluded rather than resolved with priority rules.

## Population and sampling

Fifty stocks and ETFs cover January 2005 through September 2026, with IPO-limited histories for newer names. Prices are split/dividend-adjusted Yahoo Finance OHLCV; raw inputs and hashes are managed by the shared downloader.

All twenty visible sessions must fit within the assigned split:

- Training chart end on or before December 31, 2018.
- January through March 2019 is embargoed.
- Test chart start on or after April 1, 2019.

Sampling uses seeded hash ordering and round-robin allocation across ticker/year cells separately for each class and split. It requests 75 examples per cell of the class/split table, never relaxes pattern definitions to fill deficits, and does not cherry-pick visually attractive examples. The balanced class frequencies do not estimate natural-market prevalence.

| Class | Available train | Available test | Selected train | Selected test |
|---|---:|---:|---:|---:|
| Doji | 21,710 | 12,317 | 75 | 75 |
| Hammer | 1,148 | 875 | 75 | 75 |
| Shooting Star | 884 | 672 | 75 | 75 |
| Bullish Engulfing | 5,812 | 3,069 | 75 | 75 |
| Bearish Engulfing | 6,541 | 3,771 | 75 | 75 |
| Morning Star | 401 | 189 | 75 | 75 |
| Evening Star | 334 | 226 | 75 | 75 |
| None | 124,183 | 70,853 | 75 | 75 |

**Rare classes:** Evening Star has 560 eligible single-label examples and Morning Star 590. Neither has a sampling deficit. Selected class/split subsets span 36 to 42 tickers and all 14 train years or 8 test years. Exact strata appear in [sampling_strata.csv](results/sampling_strata.csv).

Across **254,240 eligible windows**, the complete detector flag sets agree in every case. Per-class positive Jaccard agreement (both positive divided by either positive) is **100% for every class**. The pipeline excludes 1,255 multiple-pattern windows and 3,840 windows falling in or crossing the embargo. [agreement.json](results/agreement.json) reports positive counts and denominators, as well as ordinary binary agreement.

This is independent implementation of the same rulebook, **not independent expert ground truth**. Perfect numerical agreement does not establish that all tiny shapes are visually distinguishable or that the labels have trading value.

## Samples and visual audit

Nine sample charts, selected by a fixed seed before visual inspection, are in [results/samples](results/samples). Their [manifest](results/samples/manifest.json) maps anonymous sample images to labels and source dates. The nine comprise one per class plus a second Doji.

The 50-chart audit uses fixed stratification across all eight classes and both splits. Audit sheets have captions outside the input images; captions are never supplied to the model. [hand_review.json](results/hand_review.json) records each reviewed ID, numerical diagnostics, visual status and notes. The review is an AI-assisted visual inspection, not an independent human expert annotation. All 50 were inspected: 40 were visually consistent and 10 were flagged as subtle or ambiguous, with no obvious rendering defects or contradictions with the frozen numerical definitions. Ambiguous cases remain in the dataset and are flagged rather than silently replaced. This is not an estimated human label error rate.

## Reproduce

From the repository root, after installing the shared pinned environment and downloading the data:

```sh
06-jev-patterns/.venv/bin/python 06-jev-patterns/06a-candles/src/detect.py
06-jev-patterns/.venv/bin/python -m unittest discover -s 06-jev-patterns/tests -p test_candles.py -v
```

`--labels-only` runs the detector and sampling stages without rendering. `results/index.csv` is the scoring/baseline index. Each row includes an anonymous relative PNG path and a `bars_file` containing exactly the twenty visible OHLCV bars; labels and dates are separate metadata. Full charts are in ignored `charts/`, numeric bars in ignored `data/bars/`.

[render_manifest.json](results/render_manifest.json) pins rules, code, context, input-manifest and index hashes and runtime versions. [chart_hashes.json](results/chart_hashes.json) pins every generated chart. Regeneration invalidates the visual-audit status until the regenerated charts are reviewed again. Yahoo may revise historical adjustments, so exact reproduction uses the preserved local raw snapshot and its hashes.

Ten tests cover every class, trend rejection, gap/penetration rejection, strict engulfing boundaries, overlap exclusion, detector parity on randomized bars, causality, date boundaries, stratification, and the difference between positive agreement and inflated all-negative agreement. Shared rendering and metric tests run separately in CI.

Day 2 needs the real multi-bar pattern detector dataset, its review, and the final fixed inference-arm manifests. JEV scoring, repeatability parity, definitions/zero-shot/placebo/style arms and baseline training remain separate steps under the combined plan.
