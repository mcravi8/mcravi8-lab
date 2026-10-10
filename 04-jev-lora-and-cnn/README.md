# Experiment 04: JEV LoRA fine-tune vs CNN baseline

## Question
Experiment 03 found no edge for zero-shot JEV on weekly charts of 12 tickers. Does task-specific training help? We fine-tune JEV-27B-VL with a LoRA adapter and train a small CNN from scratch on the same charts, then compare both to 03's zero-shot and calibrated JEV.

## Design
- Data, charts, splits, labels and prompt identical to 03 (12 tickers, weekly 5-day charts, Up/Flat/Down next week). Test period 2025-10-17 to 2026-10-02: 492 charts (41 weeks x 12 tickers).
- JEV LoRA (`src/lora_train.py`, run on a Runpod H100 80GB HBM3): System 1 decision LoRA merged into the base, new LoRA rank 16 (108.8M trainable params), lr 1e-4, grad accumulation 8, cross-entropy over the A/B/C choice-token logits through the same readout as /v1/decide. Validation = last ~20% of train weeks; early stopping on validation log loss, evaluated twice per epoch.
- CNN (`src/cnn.py`): small CNN trained from scratch, 5 seeds; reported as the 5-seed probability-averaged ensemble and per seed.
- Evaluation (`src/evaluate04.py`, reusing 03's `src/evaluate.py` unchanged): hit rate vs each ticker's train-majority class, 95% week-block bootstrap CIs (2000 resamples) pooled, mirrored-chart placebo hit rate, Brier vs train class frequencies, Brier skill score (BSS), share of charts predicted Flat. Calibrated JEV applies 03's train-fit calibration.
- Success criteria (same as 03): hit-rate CI above test-majority base, hit > placebo, BSS > 0.
- Compute: pod ran about 12:45 to 1:26 PM PT on 2026-10-10 (~41 min at $3.99/hr, ~$2.70). LoRA training itself took 832 s.

## Training curves
![training curves](results/training_curves.png)

JEV LoRA validation log loss (ln 3 = 1.0986):

| epoch | train loss | val log loss | val acc |
|---|---|---|---|
| 0.0 | - | 1.5067 | 0.365 |
| 0.5 | 1.1652 | **1.0991** | 0.378 |
| 1.0 | 1.1088 | 1.1336 | 0.282 |
| 1.5 | 1.0827 | 1.1358 | 0.365 |

Best checkpoint is epoch 0.5, whose validation log loss is essentially uniform (1.0991 vs 1.0986). Train loss kept falling while validation got worse: the adapter first removed zero-shot JEV's overconfidence and then overfit.

## Results (pooled, 492 test charts)
Train-majority base rate 0.370; test-majority base rate 0.396; Brier reference 0.666.

![model comparison](results/model_comparison.png)

| model | hit | edge vs train base [95% CI] | placebo hit | Brier | BSS [95% CI] | % Flat |
|---|---|---|---|---|---|---|
| JEV zero-shot (03) | 0.372 | +0.002 [-0.059, +0.063] | 0.374 | 0.878 | -0.318 [-0.427, -0.212] | 0.4% |
| JEV calibrated (03) | 0.358 | -0.012 [-0.071, +0.047] | 0.362 | 0.661 | +0.007 [-0.012, +0.029] | 0.6% |
| JEV LoRA (04) | 0.376 | +0.006 [-0.041, +0.053] | 0.386 | 0.662 | +0.006 [-0.007, +0.020] | 0.0% |
| CNN 5-seed ensemble (04) | 0.372 | +0.002 [-0.043, +0.051] | 0.378 | 0.664 | +0.002 [-0.012, +0.015] | 0.0% |

CNN per seed (pooled):

| seed | hit | edge [95% CI] | placebo hit | Brier | BSS | % Flat |
|---|---|---|---|---|---|---|
| 0 | 0.337 | -0.033 [-0.077, +0.014] | 0.388 | 0.669 | -0.004 | 21.3% |
| 1 | 0.364 | -0.006 [-0.081, +0.069] | 0.392 | 0.666 | +0.000 | 0.0% |
| 2 | 0.380 | +0.010 [-0.039, +0.059] | 0.382 | 0.664 | +0.003 | 0.0% |
| 3 | 0.339 | -0.030 [-0.083, +0.026] | 0.350 | 0.668 | -0.004 | 44.5% |
| 4 | 0.380 | +0.010 [-0.039, +0.059] | 0.380 | 0.663 | +0.004 | 0.0% |
| mean (sd) | 0.360 (0.021) | -0.010 | 0.378 | 0.666 | -0.000 (0.004) | 13.2% |

Predicted class counts (Up/Flat/Down): zero-shot 254/2/236, calibrated 289/3/200, LoRA 490/0/2, CNN ensemble 486/0/6.

Per ticker (41 charts each, i.i.d. bootstrap CIs as in 03): no model has a ticker CI excluding zero. LoRA and the CNN ensemble match the base rate exactly (CI [0, 0]) on most tickers because they predict Up almost always. The largest edges are on NVDA, whose train majority was not Up (base 0.244): LoRA +0.146 [-0.098, +0.390], CNN +0.122, calibrated JEV +0.171; these come from predicting Up in an Up-heavy test year, not from chart reading. Full per-ticker numbers are in `results/metrics.json`.

## Success criteria
No model meets them. All four hit-rate CIs include the test-majority base (upper bounds +0.008 to +0.030 vs test base). LoRA and CNN are at or below their placebo (LoRA hit minus placebo CI [-0.020, -0.004], i.e. significantly below placebo). BSS CIs all include zero; only zero-shot is clearly negative.

## Limitations
- One test year, 41 weeks; CIs of about +/-5 points pooled and +/-15 to 25 points per ticker.
- LoRA: single run, single hyperparameter setting, best at a half-epoch checkpoint; no seed variance estimate. The recipe is our own (the model card documents none).
- Mirrored placebo for LoRA and CNN is not a strong control when a model predicts one class almost always.
- Same 12 tickers and labeling as 03; results do not generalize beyond that setup.

## Conclusion
Training on the charts did not create an edge. Both the LoRA-tuned JEV and the CNN converged to near-uniform probabilities that almost always pick Up, which repairs zero-shot JEV's poor Brier score (BSS -0.318 to about 0) but carries no information beyond base rates. Combined with 03, there is no evidence that weekly price charts of these tickers contain a signal these models can learn.
