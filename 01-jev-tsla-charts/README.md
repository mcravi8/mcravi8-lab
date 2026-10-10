# Experiment 01: JEV-27B-VL on TSLA charts

## Question

Can JEV-27B-VL, zero-shot, predict next-five-trading-day direction (Up, Flat, Down) from unlabeled 20-day TSLA candlestick charts better than the majority-class base rate? Does it do better on TSLA than NVDA?

## Status

Stages 0, 1, and 2 are complete: repository scaffold, daily data download, and chart rendering. The pipeline stops here before GPU provisioning or model inference. The numbers below describe data and labels, not model performance.

## Reproduce Stages 1 and 2

Use Python 3.11 and run from this experiment directory:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/data.py --as-of 2026-10-09
.venv/bin/python src/render.py
.venv/bin/python -m unittest discover -s tests -v
```

`src/data.py` without `--as-of` defaults to the current New York date. The start defaults to 2024-01-01. Since yfinance's end date is exclusive, the script requests the next day. It conservatively omits an in-progress current session until 16:15 New York time. The [yfinance API documentation](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html) defines the date and adjustment options.

Daily OHLC are explicitly adjusted (`auto_adjust=True`) so NVDA's split is handled consistently for charts and returns. Volume is the share count returned by Yahoo. Missing values, duplicate dates, nonfinite values, and malformed OHLCV stop the pipeline instead of being filled. Direct dependency versions are pinned in `requirements.txt`. All local caches are kept in this experiment directory.

## Chart and label method

Each image contains exactly 20 trading sessions of candlesticks plus volume. It is a 640 x 480 RGB PNG with a white background and no axes, ticks, dates, prices, ticker, labels, or text metadata. Price and volume scales depend only on the visible window. Chart filenames are anonymous sequential IDs.

Let `t` be the last visible session. The image uses rows `t-19` through `t`; the target is `r = close[t+5] / close[t] - 1`. The Flat threshold uses only completed historical five-session close returns:

```python
FLAT_BAND = 1 / 3
past_5d = close.pct_change(5, fill_method=None)
sigma = past_5d.rolling(60, min_periods=60).std(ddof=1)
threshold = FLAT_BAND * sigma.iloc[t]
```

The named constant `FLAT_BAND = 1 / 3` makes the Flat band one-third of the trailing five-session return standard deviation over the 60-session lookback. Flat means `abs(r) < threshold`. Otherwise the sign of `r` determines Up or Down; equality at the threshold is Up or Down. Undefined or zero volatility fails rather than inventing a label. The 60-return sample standard deviation needs 65 observed closes. The five-session stride is anchored to the first complete 20-session image (end position 19); warmup candidates are skipped until end position 64, and candidates without five future sessions are omitted. Only the past enters the image and its scales; future prices are used only for the target in the external index.

## Data and label results

Run cutoff: 2026-10-09. Each ticker has 696 daily sessions from 2024-01-02 through 2026-10-09. Eligible chart end dates run from 2024-04-04 through 2026-10-01.

| Ticker | Up | Flat | Down | Total charts | Majority class | Base rate |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| TSLA | 45 | 41 | 40 | 126 | Up | 35.71% |
| NVDA | 58 | 30 | 38 | 126 | Up | 46.03% |
| Combined | 103 | 71 | 78 | 252 | Up | 40.87% |

The majority-class base rate is the largest label count divided by the ticker's total chart count: the hit rate obtained by always predicting that class. For TSLA it is 45/126; for NVDA it is 58/126.

The brief's approximate 200 charts per ticker is larger than the exact date range and five-session stride allow. This run preserves those settings, the full volatility warmup, and the five-session outcome requirement, producing 126 per ticker.

Generated local inputs (ignored by Git):

- `data/TSLA.parquet` and `data/NVDA.parquet`: validated daily OHLCV.
- `data/manifest.json`: requested dates, actual sessions, download timestamp, yfinance version, and Parquet checksums.
- `charts/chart_*.png`: all 252 anonymous input charts.
- `charts/index.csv`: exactly `file,ticker,end_date,label,fwd_ret`; file paths are relative to this experiment directory.

Committed Stage 2 outputs:

- `results/label_counts.json`: per-ticker and combined label counts.
- `results/render_manifest.json`: rendering settings, input checksums, and chart date ranges.
- `results/samples/manifest.csv`: provenance, true labels, returns, and checksums for the three samples.

Samples are selected by fixed temporal positions without inspecting their labels: TSLA quarter, NVDA midpoint, TSLA three-quarter, using `floor((n-1) * fraction)`. They are copied byte-for-byte from the corresponding model-input PNGs. Captions below are documentation; the images contain no labels.

| Sample | Ticker | Chart end date | True next-five-day label |
| --- | --- | --- | --- |
| 1 | TSLA | 2024-11-13 | Flat |
| 2 | NVDA | 2025-07-01 | Up |
| 3 | TSLA | 2026-02-11 | Down |

![Sample 1](results/samples/sample_01.png)

![Sample 2](results/samples/sample_02.png)

![Sample 3](results/samples/sample_03.png)

## Validation

Five tests cover the hand-calculated trailing sample standard deviation, the strict Flat boundary, the 20-session window and five-session stride/horizon, input validation, and future-price mutations. Changing later OHLCV leaves historical volatility and the earlier chart PNG bytes unchanged. Real-data labels were independently recomputed with explicit loops and matched all 252 index rows. The generated images and sample provenance were inspected.

## Limitations and next steps

Yahoo's adjusted history is retrospective and can be revised; it is not a point-in-time archive. Checksums identify this run's local data snapshots. Adjacent 20-session images overlap. The experiment is zero-shot, uses only TSLA and NVDA, and does not study trading P&L. Fine-tuning and extra tickers are out of scope.

Stage 3 will serve `autotrust/JEV-27B-VL` on one 80 GB A100/H100 via vLLM per the model card, using System 1 decision mode with options ["Up", "Flat", "Down"] and the prompt: "This is a 20-day daily stock chart. Over the next 5 trading days, will the price go Up, stay Flat, or go Down?" Scoring must be resumable and batched, with model weights cached on a RunPod network volume. Keep future HF tokens and RunPod keys only in an ignored .env file.

Later evaluation will compare argmax hit rate and Brier score with the class-frequency baselines, bootstrap 95% hit-rate intervals with 1,000 resamples, plot calibration, and compare TSLA with NVDA. The brief defines model success only when the TSLA hit-rate confidence interval excludes the base rate. Planned sanity checks score horizontally flipped charts as a placebo and ask the model to identify the ticker on 20 charts. The brief reports that the model card's image calibration is unmeasured; verify this against the card before GPU scoring. The final stage will produce the three-panel LinkedIn figure and an honest model-results write-up.
