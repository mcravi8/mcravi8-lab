# Experiment 02: zero-shot JEV-27B-VL on weekly 5-day charts

## Question
Can JEV-27B-VL, with no fine-tuning, look at an unlabeled candlestick chart of one full trading week (Monday to Friday) for TSLA or NVDA and predict whether the next week's move is Up, Flat, or Down better than simple baselines?

## Design
- Data: TSLA and NVDA adjusted daily OHLCV from yfinance, 2024-01-02 to 2026-10-09 (696 sessions each). `src/data.py` is copied from Experiment 01.
- Charts (`src/render.py`): one chart for each calendar week with exactly 5 sessions, Monday to Friday. Holiday-shortened weeks get no chart. Same style as 01: 640x480 candlesticks plus a volume panel, with no axes, dates, prices, or ticker.
- Target: r = close of the last session of the next calendar week / this Friday's close - 1. Usually that session is the next Friday. If the next week is holiday-shortened, its last session is used instead (22 of 105 charts per ticker).
- Labels: Flat if |r| < (1/3) x sigma, otherwise Up or Down by sign. sigma is the sample std (ddof=1) of the 12 most recent calendar-week returns, ending with the charted week, so it uses only data known at the chart's Friday close. Weeks without 12 prior returns are skipped as warmup. The first chart ends 2024-04-05.
- Split by date, shared by both tickers: train covers chart weeks ending 2024-04-05 to 2025-10-03 (64 per ticker). The week ending 2025-10-10 is dropped as a gap. Test covers 2025-10-17 to 2026-10-02 (41 per ticker), about 39% of the sample.
- Model: JEV-27B-VL System 1 `/v1/decide`, served from the model snapshot with the decision LoRA. Request: kind=choice, options Up/Flat/Down, thinking off. Question: "This is a 5-day daily stock chart covering one trading week. Over the next week, will the price go Up, stay Flat, or go Down?" The only change from 01's `jev_api.py` is this question.
- Placebo: the same charts mirrored left to right (`score.py --placebo`).
- Metrics (`src/evaluate.py`): argmax hit rate and two majority-class base rates. The test-majority rate looks ahead; the train-majority rate applied to test is the honest one. Ties break in Up, Flat, Down order, which matters for NVDA, where train Flat and Down tie at 22. Also a 95% bootstrap CI of hit rate minus base rate (2000 resamples, seed 0), placebo hit rate, multiclass Brier of the model vs. train class frequencies, and BSS = 1 - Brier_model / Brier_ref. Success requires all three: CI above zero, beats placebo, and BSS > 0.

## Results (holdout is primary)
Label counts, train / test (Up-Flat-Down): TSLA 24-20-20 / 15-14-12, NVDA 20-22-22 / 16-10-15.

| Ticker | Sample | n | Hit rate | Base (test maj.) | Base (train maj.) | 95% CI hit - base (test maj.) | Placebo | Brier model | Brier ref | BSS |
|---|---|---|---|---|---|---|---|---|---|---|
| TSLA | holdout | 41 | 0.439 | 0.366 (Up) | 0.366 (Up) | [-0.098, 0.244] | 0.415 | 0.816 | 0.665 | -0.226 |
| NVDA | holdout | 41 | 0.317 | 0.390 (Up) | 0.244 (Flat) | [-0.244, 0.098] | 0.439 | 0.983 | 0.671 | -0.465 |
| TSLA | full | 105 | 0.400 | 0.371 (Up) | 0.371 (Up) | [-0.076, 0.133] | 0.352 | 0.874 | 0.665 | -0.316 |
| NVDA | full | 105 | 0.333 | 0.352 (Down) | 0.305 (Flat) | [-0.133, 0.095] | 0.333 | 0.923 | 0.668 | -0.383 |

The CI against the train-majority base: TSLA holdout [-0.098, 0.244], NVDA holdout [-0.146, 0.293].

The model never predicted Flat by argmax. Its mean Flat probability was about 0.09 to 0.11, even though Flat is about a third of the labels. This is the main reason its Brier score is worse than the class-frequency reference.

**Success criteria: not met for either ticker.** All CIs include zero and BSS is negative for both. TSLA beats its placebo, but that difference is inside the sampling noise. NVDA does not beat its placebo. Full numbers are in `results/metrics.json`.

## Limitations
- Small samples: 41 test charts per ticker, so the CIs are about +/-0.17 wide.
- Zero-shot only, with a single prompt. Other prompt wording was not tested.
- The Flat band is defined in terms of volatility, and JEV has no notion of it, which helps explain why it never chose Flat.
- Charts are rescaled per week, so the absolute size of the move is not visible.
- Weekly returns overlap with holiday-shortened target weeks. Labels use adjusted closes from yfinance as downloaded on 2026-10-10.

## Reproduce
```sh
PY=../01-jev-tsla-charts/.venv/bin/python
$PY src/data.py && $PY src/render.py
JEV_BASE_URL=https://<POD_ID>-8000.proxy.runpod.net $PY src/score.py --api decide
JEV_BASE_URL=... $PY src/score.py --api decide --placebo
$PY src/evaluate.py
```
Serving uses the same setup as 01's README (`scripts/remote_serve.sh` with MODEL_DIR=/workspace/models/JEV-27B-VL, PORT=8000, MAX_MODEL_LEN=8192). Run 2026-10-10 on one H100 80GB HBM3 pod (Runpod US-MO-1, $3.99/hr) for 10 min 40 s, about $0.71. A100 80GB was unavailable.
