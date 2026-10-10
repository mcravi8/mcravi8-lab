# Experiment 01: JEV-27B-VL on TSLA charts

## Question

Can JEV-27B-VL, zero-shot, predict next-five-trading-day direction (Up, Flat, Down) from unlabeled 20-day TSLA candlestick charts better than the majority-class base rate? Does it do better on TSLA than NVDA?

## Method

- Download daily OHLCV for TSLA and NVDA from 2024-01-01 through the run date with yfinance.
- Render 20-trading-day candlestick and volume windows every five trading days. Images contain no axes, ticks, dates, prices, ticker, or labels.
- Define the forward return as close[t+5] / close[t] - 1. Label Flat when its absolute value is strictly less than 0.5 times the trailing standard deviation of five-day returns over 60 observations; otherwise label Up or Down by the return's sign. Use only completed historical returns in that threshold.
- In the later GPU stage, use autotrust/JEV-27B-VL System 1 decision mode with options ["Up", "Flat", "Down"] and the prompt: "This is a 20-day daily stock chart. Over the next 5 trading days, will the price go Up, stay Flat, or go Down?"
- In later evaluation, compare argmax hit rate and Brier score with the class-frequency baselines, bootstrap 95% hit-rate intervals with 1,000 resamples, plot calibration, and compare tickers. The model works under the brief's criterion only when the TSLA hit-rate confidence interval excludes the base rate.

## Setup

Python 3.11. Run from this experiment directory:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/data.py
```

The download defaults to 2024-01-01 through the current New York date. For a fixed cutoff, use `src/data.py --as-of YYYY-MM-DD`. Since yfinance's end date is exclusive, the script requests the next day. It omits an in-progress current session until 16:15 New York time. The [yfinance API documentation](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html) defines the date and adjustment options. Daily OHLC are explicitly adjusted (`auto_adjust=True`) so NVDA's split is handled consistently. Volume is the share count returned by Yahoo. Missing values, duplicate dates, and malformed OHLCV stop the download instead of being filled. `data/manifest.json` records the data range, package version, and Parquet checksums.

Keep any future HF token and RunPod key in an ignored .env file.

## Status and results

Stage 1 complete. Downloaded 696 daily sessions per ticker, from 2024-01-02 through 2026-10-09. No model predictions or performance results yet.

## Limitations and planned checks

The experiment is zero-shot, uses only TSLA and NVDA, and does not study trading P&L. Adjacent image windows overlap. Later checks will score horizontally flipped charts as a placebo and ask the model to identify the ticker on 20 charts. The brief reports that the model card's image calibration is unmeasured; verify that statement against the card before GPU scoring.

## Next steps

Complete data download and chart rendering, publish three sample charts and label counts, then stop before provisioning a GPU or running model inference. Fine-tuning and extra tickers are out of scope.
