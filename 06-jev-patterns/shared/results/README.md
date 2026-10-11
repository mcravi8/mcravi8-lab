# Experiment 06 results and repeatability noise floor

No v6 GPU inference has run. This report analyzes the user-supplied paired candle gates on CPU; [raw failures and hashes](diagnostics/2026-10-10-v5-parity/diagnosis.json) and [recomputed statistics](parity_noise_floor.json) are preserved.

## Historical candle noise floor

Each measured session scored the same 20 charts twice with eight ordered options. Mean/max/p90 below describe **160 absolute option-probability differences**, including all flagged charts. P90 uses linear interpolation. The additional chart-level p90 describes the 20 per-chart maximum differences. Values use probability units, not percentages.

| Recorded session | Mean option drift | Max option drift | P90 option drift | P90 chart-max drift | Argmax match | Charts >0.03 |
|---|---:|---:|---:|---:|---:|---|
| Original v2 (called v3 in user request) | 0.00197762 | 0.03334437 | 0.00561157 | 0.01391830 | 20/20 | #15 |
| v3 startup | N/A | N/A | N/A | N/A | N/A | No decide calls |
| v4 | 0.00202602 | 0.01841074 | 0.00597012 | 0.01470235 | 20/20 | None |
| v5 | 0.00245812 | 0.04121730 | 0.00572323 | 0.01828424 | 20/20 | #15 |

Mean drift is roughly **0.20–0.25 percentage points**, with observed worst-case drift as high as **4.12 percentage points**. All top predictions agreed. The recurring flagged chart is `charts/b5674ffc8efb3bc45e6d.png`. The original and v5 scorer metadata record the same PNG SHA-256: `0aed05d7b75aa6e4b93166f53e9e7a7544bb9c7ae6a00b9785c6b62caa492b70`. Its filename, one-based gate position 15 and bytes match; the current image also matches. The other 19 charts stayed below 0.03 in every measured session.

The original 0.033344 evidence is stored as v2. Actual v3 failed before readiness because batch invariance was unsupported for GDN_ATTN; it has no paired predictions. Original v2/v4/v5 failures are unchanged.

## Approved v6 rule and interpretation

Matteo approved this rule **after seeing v5 fail**: **all 20 top predictions must match AND mean absolute option-probability drift must be <=0.01**. Mean is the sum of all 20 × K absolute differences divided by 20 × K; K is the declared option count. Flagged charts remain included and must also match argmax. Reports log mean, max, p90, chart-level p90 and all chart maxima above0.03. Ties select the first declared option. A flag alone does not fail this approved gate.

This replaces v5's max <=0.03 criterion; the original criterion was max <=0.01. The reason is residual BF16 kernel nondeterminism under unchanged serial batch1, eager, no-prefix-cache/no-chunked-prefill settings, with one recurring tail outlier and stable classifications. The exact kernel/build cause has not been isolated.

**The noise floor is an observed repeatability result**, scoped to this fixed candle fixture and these builds. It is not an independent validation, a universal bound, a pattern-task gate, HF-vLLM equivalence or evidence of recognition accuracy. The policy was adapted using these observations. Small changes in probabilities or calibration on a similar scale need interpretation alongside this variation; all later results must disclose the policy history and observed noise floor. Every future pending arm retains its live two-pass gate. CPU replay of historical pairs passes v6 but does not replace a fresh GPU gate.

## Other reported CPU results

The real-pattern [detector-agreement table](../../06b-chart-patterns/results/detector_agreement.csv) is a reported CPU result. All 150 real-pattern parents remain showcase-only; no aggregate or per-class model accuracy claims are permitted. See the [runbook](../../GPU_RUNBOOK.md) for the unchanged 9,820-call frozen schedule and pending GPU steps.
