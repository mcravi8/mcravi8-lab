# Experiment 06 (combined): Can JEV-27B-VL read candlestick and chart patterns? (PLANNED, not run)
Sub-parts: 06a candlestick patterns, 06b multi-bar chart patterns. Detailed earlier notes: PROPOSAL_06a_candles_v1.md, PROPOSAL_06b_chart_patterns.md.

## Shared infrastructure (build once, Day 1 morning)
- data.py: yfinance daily OHLCV, ~50 large-cap US stocks + ETFs (includes Exp 03's 12), 2005 to 2026.
- render.py: Exp 01 no-label style, parameterized: n_bars (20 for candles, 60 to 120 for chart patterns), size (640x480 / 1024x640), style (candle, OHLC bar, line), mirror flag.
- score.py + jev_api.py: Exp 03 decide client generalized to any option list and an optional context prefix; resumable; parity check (same 20 charts scored twice must match within 0.01).
- evaluate.py: accuracy, macro-F1, confusion matrix, Brier, ECE (15 bins), reliability diagram, risk-coverage (p >= 0.5/0.7/0.9), block bootstrap CIs (ticker-quarter), arm comparison (zero-shot vs context).
- baselines.py: majority/uniform, small CNN (Exp 04 arch), logistic on numeric bars.
- Tests: label fixtures per pattern, split-boundary asserts, metric fixtures (CI from Exp 05).

## 06a Candlestick patterns
- Classes (8): Doji, Hammer, Shooting Star, Bullish Engulfing, Bearish Engulfing, Morning Star, Evening Star, None.
- Labels: TA-Lib CDL* on last candle (+ prior 5-day trend where required); second detector (pandas/CandleKit) agreement; exclusive single label; 150 per class (1,200 charts); hand-check 50.
- Render: 20 candles + volume, pattern at right edge.
- Context: ~150 words, TA-Lib thresholds in plain terms.

## 06b Multi-bar chart patterns
- Classes (6 + None): H&S, Inverse H&S, Double Top, Double Bottom, Triangle (merged), Bull Flag, None. Deferred: Cup and Handle, VCP, harmonics.
- Part 1 synthetic: generator for clean shapes + noise ladder (5 noise levels), 200 clean + 1,000 noisy, exact labels.
- Part 2 real: BennyThadikaran/stock-pattern (verify list/license) + own zigzag detector agreement; completed by last bar; 80 to 120 per class (~700); hand-check 10 to 15 per class.
- Render: pattern span + 10 to 20% padding (60 to 120 bars) at 1024x640; also fixed 100-bar framing.
- Context: ~250 words geometric definitions; optional 1 reference sketch per class.

## Splits
Date-based for real data: train <= 2018, 3-month gap, test >= 2019 (both sub-parts, so one rule). Train used only for CNN, logistic, calibration. Synthetic set has no split (eval only).

## Arms (both sub-parts)
zero-shot, definitions, (optional) definitions + reference images; plus mirrored placebo (06a: position control; 06b: expected class flips H&S <-> Inverse, Top <-> Bottom) and style variants (candle / OHLC / line) on a 200-chart subset.

## Timeline (~2.5 days)
- Day 1 AM: shared infra + tests. Day 1 PM: 06a labels/render, 06b synthetic generator.
- Day 2 AM: 06b real detector labels + hand review (the slow part). Day 2 PM: single GPU session scores everything (all arms, placebo, styles), stop pod.
- Day 3 AM: evaluation, figures, READMEs, push.
Fallback if Day 2 slips: ship 06a + 06b synthetic first, real 06b later.

## Folder
~/Desktop/WORK/ai-finance-experiments/06-jev-patterns/
  shared/src/ (data, render, jev_api, score, evaluate, baselines), tests/
  06a-candles/ (src/detect.py, results/, README)
  06b-chart-patterns/ (src/detect_patterns.py, synth.py, results/, README)
  README.md (combined findings)
Plan notes: ~/Desktop/GROKsBOT/scout/experiments/06-jev-candle-patterns/

## LinkedIn angle (one post, paired findings)
"Can an AI read charts like a trader? I tested it on 8 candlestick patterns and 6 classic chart patterns." Paired story: small local shapes (candles) vs large global shapes (H&S, triangles); with vs without the rulebook; where it breaks as noise rises; whether its confidence can be trusted. Visuals: side-by-side confusion matrices, accuracy-vs-noise curve, a grid of example charts with model confidence. Ties back to the price-prediction work: it may not predict prices, but can it see what traders see?

## Risks and controls
- Label subjectivity (esp. 06b): two-detector agreement, hand review, report high-agreement subset.
- Framing leak (06b variable windows): fixed 100-bar control.
- Barely visible patterns (06a): report on clear-cut subset.
- Prompt sensitivity: 2 paraphrases per arm.
- Inference parity: enforced check before scoring (Codex Exp 04 finding).
- Holdout reuse: freeze specs before scoring; test set scored once.
- GPU availability: A100 often sold out in US-MO-1; H100 fallback.

## Cost
GPU: ~6,000 to 9,000 decide calls (06a 1,200 + 06b ~1,900, x 2 to 3 arms, + placebo/style), ~1 to 1.5 h, about $2 to $6 (A100 $1.79/h or H100 $3.99/h). CPU: ~2 days. No other spend.
