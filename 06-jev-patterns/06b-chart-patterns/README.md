# Experiment 06b: synthetic chart-pattern recognition

This CPU dataset prepares a visual-recognition test for **[autotrust/JEV-27B-VL on Hugging Face](https://huggingface.co/autotrust/JEV-27B-VL)**. No model inference, GPU session, or Runpod resource is started. Real-market pattern detection remains Day 2 work.

Protocol **v3**, seed **60618**, replaces the original generator's overly simple None paths and deterministic nuisance cues. Its [parameters and geometric rules](protocol.json) record the initial revision, a failed intermediate classifier check, and the final negative-family design fixed before a second, freshly seeded confirmation check. The 200 parent templates each receive five paired corruptions, producing 1,200 images at 1024 × 640 with 60–120 candlesticks and volume. Scored images contain no text.

| Source class | Clean parents | Noisy variants | Total |
|---|---:|---:|---:|
| Head and Shoulders | 29 | 145 | 174 |
| Inverse Head and Shoulders | 29 | 145 | 174 |
| Double Top | 29 | 145 | 174 |
| Double Bottom | 29 | 145 | 174 |
| Triangle | 28 | 140 | 168 |
| Bull Flag | 28 | 140 | 168 |
| None | 28 | 140 | 168 |
| **All** | **200** | **1,000** | **1,200** |

None now comprises **5 random walks, 5 choppy ranges, 5 trends with random pullbacks, 5 incomplete formations, 4 broken reversals, and 4 broken consolidations**. The latter families include triple tops/bottoms and unequal double peaks/troughs, diverging bounds, and rising consolidations that fail the defined downward Bull Flag rule. They can have completed-looking exits, so an absent breakout is not a universal negative cue. This six-family mixture was fixed for coverage before its classifier results. A frozen screen rejects obvious completed whole-window target shapes in their source control points; the primary dataset rejected one candidate path. The screen is deliberately limited to synthetic control paths, and does not implement Day 2's real-data detector or establish that every image lacks an incidental local pattern. Acceptance never uses the leakage classifier's output.

Every family uses the same independent distributions for bar count, nominal price, target volatility, microtexture, candle gaps/wicks, and volume. Each clean parent's sample standard deviation of daily **log returns** is scaled exactly to an independent log-uniform draw between 0.004 and 0.018. This matches the population distribution rather than assigning identical finite-sample values to every class. [Population statistics](results/population_stats.json) report the realized per-class ranges and medians, lengths, positions, and margin directions.

Positive patterns vary pivot timing, height, neckline tilt, shoulder asymmetry, channel width/slope, and triangle subtype within their geometric definitions. Shared daily microtexture is added to every clean family, including None. Pattern location and both margins vary independently: each margin spans 10–20% of the pattern length, and its direction is unrelated to the class or preceding segment. A separately seeded volume process uses **neutral gray bars**, so volume colors cannot duplicate the price direction. Clean positive parents retain a completed break and stay beyond it through the right margin.

“Clean” means **no added corruption**; it does not mean a perfectly smooth line. The five added levels are **0.035, 0.08, 0.16, 0.30, and 0.55** times the clean log-price range. Each parent reuses one global Brownian bridge, pinned only at the chart endpoints. Major pivots and even apparent completion can disappear at high noise. Labels identify the source family, not a claim that a severely corrupted picture still clearly exhibits it. All six versions retain `template_id` and remain together in confidence intervals or paired analyses.

The [noise grid](results/samples/synthetic_grid.png) shows the first parent of every class at levels 0, 0.08, and 0.55. The [clean-diversity grid](results/samples/clean_diversity_grid.png) shows the first three independently drawn parents per positive class and the first six None parents, covering all six negative families in two rows. Selection rules were fixed before viewing. The seven individual clean examples remain `results/samples/sample_01.png` through `sample_07.png`. Grid captions are review aids and never enter scored images.

![Clean and noisy synthetic patterns](results/samples/synthetic_grid.png)

![Independent clean parent diversity](results/samples/clean_diversity_grid.png)

From the repository root, using the separate pinned Experiment 06 environment:

```sh
06-jev-patterns/.venv/bin/python 06-jev-patterns/06b-chart-patterns/src/synth.py
06-jev-patterns/.venv/bin/python 06-jev-patterns/06b-chart-patterns/src/leakage_audit.py
06-jev-patterns/.venv/bin/python -m unittest discover -s 06-jev-patterns/tests -p 'test_synthetic*.py' -v
```

Exports include ignored `data/bars/*.csv`, ignored `charts/*.png`, and ignored `charts/index.csv`. Committed `results/synthetic_index.csv` duplicates the scorer index; counts, population statistics, and the render manifest preserve seeds, pivots, breaks, data/image hashes, and protocol/generator/renderer/context hashes. CSV dates are artificial coordinates required by the shared renderer: `ticker=SYNTH`, index `end_date` is blank, and no historical inference is valid.

## Numeric shortcut regression check

The 200 evaluation parents have no training split. The [CPU audit](src/leakage_audit.py) fits only separate cohorts of **1,400 clean parents**, each split into 840 training and 560 held-out parents, stratified over seven source classes. Each held-out cohort contains 80 None and 480 positive patterns. No audit parent seed overlaps the primary 200, the other audit cohort, or the opposite split; noise/style siblings never cross partitions. Inputs are only visible numeric OHLCV, with 32 normalized close positions and fixed shape/nuisance statistics.

Before inspecting revised-generator outcomes, we fixed a standardized class-balanced logistic regression (`C=1`) and a class-balanced 200-tree forest (`max_depth=4`, `min_samples_leaf=10`). Each must stay **strictly below 90% held-out balanced accuracy** for None versus any declared pattern. Balanced accuracy averages the two class recalls; ordinary accuracy would reward the 6:1 positive-class imbalance. Always guessing pattern already scores 85.71% ordinary accuracy and 50% balanced accuracy.

| Generator / cohort | Logistic balanced accuracy | Forest balanced accuracy | Forest ordinary accuracy |
|---|---:|---:|---:|
| Original generator | 100.00% | 100.00% | 100.00% |
| Intermediate v2 | 78.44% | 95.42% | 98.39% |
| Final v3 regression | 63.44% | 86.35% | 93.57% |
| Final v3 fresh confirmation | 67.50% | 88.13% | 94.82% |

The intermediate forest failure motivated the rule-defined hard negatives. [Development history](results/leakage_development_history.json) retains that failure and the original-generator result. The reused cohort (`20261017`, split `1701`) is a regression check after adaptation, not an untouched holdout. A second cohort (`20261023`, split `2301`) was declared before evaluating v3; the features, classifiers and pass limits stayed unchanged. The [final report](results/leakage_audit.json) contains both cohorts, confusion matrices, source hashes and complete configuration. Tests enforce both gates on both cohorts and include a deliberately leaky fixture that must fail.

These are conditional regression guards, **not proof of zero leakage or model quality**. Strong geometry-based separation can be legitimate; the fresh forest still has 95.44% ROC AUC. Shared nuisance distributions alone do not guarantee identical finite-sample realizations. No real JEV inference or pattern-baseline training on the primary evaluation set has run.

The prepared fixed-100 control interpolates each same template/noise realization onto 100 bars. Generate it in a separate ignored directory:

```sh
06-jev-patterns/.venv/bin/python 06-jev-patterns/06b-chart-patterns/src/synth.py \
  --fixed-bars 100 --output-dir 06-jev-patterns/06b-chart-patterns/data/fixed_100
```

The primary Day 1 set does not generate this optional control. Day 2 real-data framing must use genuine fixed historical windows instead. Shared rendering also supports OHLC/line variants and horizontal mirrors. A horizontal mirror reverses **time**, not prices; it does not exchange H&S with Inverse or Double Top with Double Bottom, contrary to the preserved planning note. Price inversion creates those counterparts.

Day 2 still needs verified real-detector capabilities/license, an independent zigzag implementation, agreement/exclusivity and completion/overlap filters, the temporal split and embargo, per-class hand review, and approximately 80–120 real examples per class where available. Freeze rules before scoring. GPU scoring requires separate authorization and a successful repeatability gate. Synthetic recognition results cannot establish market prediction or trading skill.
