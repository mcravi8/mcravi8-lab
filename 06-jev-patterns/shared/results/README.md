# Experiment 06 results and repeatability noise floor

The user-run **v7 session completed ten main-arm jobs (7,800 predictions)** and failed at the first paraphrase gate. Review and evaluation of those saved outputs use CPU only. [Main results and limitations](../../README.md#v7-results-useful-recognition-substantial-class-gaps), [v7 session summary](inference_run_v7/session_summary.json), [v6 original failures and SHA-256s](diagnostics/2026-10-11-v6-parity/diagnosis.json), [v5 evidence](diagnostics/2026-10-10-v5-parity/diagnosis.json), and [historical recomputed statistics](parity_noise_floor.json) are preserved. The frozen historical noise-floor JSON remains unchanged; the table below comes from v7 parity files.

## Recorded v7 execution and noise floor

The [server log](inference_run_v7/server.log) contains **8,320 successful HTTP 200 decide calls**: two global gates (80), ten job gates (400), the failed paraphrase gate (40) and 7,800 predictions. No HTTP failures occurred. Session elapsed time was 3,630.4 seconds (60.5 minutes); this is measured partial-run wall time, not the original full-plan GPU budget.

The predictions comprise 2,400 candle rows (both train and test, two arms), 4,800 synthetic rows (same 200 parents, six noise levels, two framings and two arms) and 600 real showcase rows (150 parents, two framings, two arms). Held-out evaluation uses only 600 test candles per arm and 1,200 synthetic variants per framing/arm. Real showcase rows cannot produce accuracy metrics.

[Candle gate files](../../06a-candles/results/inference/frozen_v7) and [pattern gate files](../../06b-chart-patterns/results/inference/frozen_v7) each score 20 charts twice. Mean/max/p90 are option-probability differences, in probability units. Near-tie counts show pass 1 / pass 2 / either pass; mismatch counts show allowed / disallowed.

| Gate | Mean drift | Max drift | P90 drift | Raw matches | Near ties 1/2/either | Mismatches allowed/disallowed | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Global candles | 0.00208658 | 0.05288545 | 0.00427483 | 20/20 | 0/0/0 | 0/0 | Pass |
| Global patterns | 0.00149169 | 0.01014593 | 0.00508691 | 20/20 | 0/0/0 | 0/0 | Pass |
| Candles zero-shot | 0.00205827 | 0.02199629 | 0.00647834 | 20/20 | 0/0/0 | 0/0 | Pass |
| Candles definitions | 0.00249554 | 0.02016069 | 0.00793826 | 19/20 | 1/0/1 | 1/0 | Pass |
| Real variable zero-shot | 0.00182186 | 0.01438924 | 0.00477303 | 19/20 | 2/1/2 | 1/0 | Pass |
| Real variable definitions | 0.00212342 | 0.02106325 | 0.00708467 | 20/20 | 0/0/0 | 0/0 | Pass |
| Real fixed-100 zero-shot | 0.00187006 | 0.00845259 | 0.00507186 | 20/20 | 0/1/1 | 0/0 | Pass |
| Real fixed-100 definitions | 0.00181774 | 0.01632896 | 0.00547502 | 20/20 | 0/0/0 | 0/0 | Pass |
| Synthetic variable zero-shot | 0.00188335 | 0.01828429 | 0.00564478 | 20/20 | 0/0/0 | 0/0 | Pass |
| Synthetic variable definitions | 0.00118959 | 0.00926313 | 0.00374666 | 20/20 | 0/0/0 | 0/0 | Pass |
| Synthetic fixed-100 zero-shot | 0.00123776 | 0.01202738 | 0.00418494 | 20/20 | 0/0/0 | 0/0 | Pass |
| Synthetic fixed-100 definitions | 0.00152655 | 0.01532507 | 0.00392431 | 20/20 | 0/0/0 | 0/0 | Pass |
| Candles paraphrase zero-shot | 0.00226674 | 0.02866155 | 0.00571889 | 19/20 | 1/1/1 | 0/1 | **Fail** |

The global candle gate flagged **chart #15, `charts/b5674ffc8efb3bc45e6d.png`**, with max drift **0.05288545 (5.29 percentage points)**. Its argmax matched, and it stayed in the overall mean. This is the recurring tail-drift file from v2/v5; the new observation exceeds the earlier 4.12-point maximum. All other v7 gates had no chart-max flags above 0.03. Across the ten main-job gates, four charts were near ties (either pass) and two changed argmaxes qualified for the exception. These gate observations can overlap and must not be counted as independent charts.

The paraphrase gate's changed chart, `charts/c3307bcc86449a1b8121.png`, moved **Shooting Star → Bearish Engulfing**. Top-two margins were **0.01154152 / 0.01155571**, both above the frozen 0.01 exception, despite mean drift 0.00226674. Its one near-tie chart was a different file with a matching argmax. The gate correctly blocked scoring; **no paraphrase prediction CSV was written**, and all style/mirror/price-inversion controls remained unrun. This tests repeatability of one wording; it provides no completed between-wording sensitivity comparison.

The approved v7 policy and all model/runtime/data settings remain unchanged. Completed main predictions are usable exploratory results with this reported noise floor; the incomplete controls prevent robustness claims. No fresh inference, pod or GPU was started for the README update.

## Historical candle noise floor

Each measured session scored the same 20 charts twice with eight options. Mean/max/p90 below describe all 160 absolute option-probability differences. P90 uses linear interpolation; chart-max p90 describes 20 per-chart maxima. Values are probability units. The original failures remain unchanged.

| Session | Mean option drift | Max drift | P90 option drift | P90 chart-max drift | Raw argmax matches | Charts max>0.03 |
|---|---:|---:|---:|---:|---:|---|
| v2 original (called v3 in earlier request) | 0.00197762 | 0.03334437 | 0.00561157 | 0.01391830 | 20/20 | #15 |
| v3 startup | N/A | N/A | N/A | N/A | N/A | No calls |
| v4 | 0.00202602 | 0.01841074 | 0.00597012 | 0.01470235 | 20/20 | None |
| v5 | 0.00245812 | 0.04121730 | 0.00572323 | 0.01828424 | 20/20 | #15 |
| v6 | 0.00260983 | 0.02224329 | 0.00769977 | 0.01945070 | 19/20 | None |

Mean drift is roughly **0.20–0.26 percentage points**, with observed max drift up to **4.12 percentage points**. Actual v3 exited before readiness because GDN_ATTN rejected batch invariance; it has no drift measurement.

## Near-tie counts under the approved v7 definition

A near-tie chart has a top-two margin<=0.01 in **either** pass. Counts include matching charts; an allowed mismatch is a separate count of changed argmaxes meeting this condition. All charts and probabilities remain in the mean.

| Session | Near ties pass 1 | Near ties pass 2 | Near ties either pass | Allowed argmax mismatches | Disallowed argmax mismatches |
|---|---:|---:|---:|---:|---:|
| v2 | 0 | 0 | 0 | 0 | 0 |
| v3 | N/A | N/A | N/A | N/A | N/A |
| v4 | 0 | 0 | 0 | 0 | 0 |
| v5 | 0 | 0 | 0 | 0 | 0 |
| v6 | 0 | 1 | 1 | 1 | 0 |

V6 gate chart #11, `charts/7d763e0bb9783f355926.png`, changed **Doji → Shooting Star**. Pass 1 probabilities were 0.4546401571 versus 0.4408969473, margin 0.0137432098; pass 2 probabilities were 0.4485482060 versus 0.4485217768, margin 0.0000264292. Only pass 2 is a near tie. Its **raw agreement remains 19/20**; the one mismatch qualifies for the v7 exception. This is a CPU replay of the original failed gate, not a new GPU pass.

The prior recurring tail-drift chart #15, `charts/b5674ffc8efb3bc45e6d.png`, has the same filename/position/hash in v2 and v5. Recorded and current SHA-256: `0aed05d7b75aa6e4b93166f53e9e7a7544bb9c7ae6a00b9785c6b62caa492b70`. Its identity proof remains in the machine-readable report.

## Approved v7 rule and interpretation

Matteo approved this final change on **2026-10-11 after seeing v6 fail**: **mean absolute chart-option drift<=0.01 AND every argmax mismatch has top-two margin<=0.01 in at least one pass**. All other charts must match. Margin is largest minus second-largest probability over all options after unchanged validation/normalization; exact ties qualify. First-declared-option tie-breaking and original probabilities/predictions remain unchanged.

This is the **third post-hoc gate change**: original max<=0.01; v5 max<=0.03 plus full agreement after v4; v6 mean<=0.01 plus full agreement after v5; v7 adds the conditional near-tie exception after v6. The reason is small BF16 probability variation reordering almost tied choices under frozen serial batch 1/eager/cacheless execution. The exact kernel/build cause has not been isolated.

Gate files, console results and job summaries report raw agreement, per-pass/either-pass near-tie counts, both top-two margins, allowed/disallowed mismatch counts, mean/max/p90 and chart-max flags>0.03. Flags and near-tie exceptions never remove charts. A disallowed mismatch or excessive mean drift blocks scoring.

The v2–v6 measurements concern one fixed 20-chart candle fixture. V7 adds the task/arm fixtures above. Neither constitutes independent validation, a general noise bound, HF-vLLM equivalence or recognition accuracy. Results must disclose the adaptations, residual noise floor and allowed mismatch counts. Small probability/calibration changes need interpretation against the observed variation. Every pending arm still requires fresh live gates.

## CPU forecasting bridge

The separate [CPU bridge results](forecast_bridge_v1/README.md) evaluate **five-session return forecasts**, using the recorded JEV pattern probability vector as features. They find no incremental benefit: held-out C-vs-A Brier reduction −0.013959 (four-retained-week 95% interval −0.024287 to −0.003695); annual walk-forward reduction −0.009699 (−0.018551 to −0.000451). C's held-out accuracy is 33.4%, versus numeric A 34.8% and training prior 39.6%. These numbers must not be confused with pattern-recognition accuracy.

All frozen continuation conditions fail. [Protocol](../../config/forecast_bridge_v1.json) and implementation were pushed before inspecting return results. The [independent audit](forecast_bridge_v1/validation.json) verifies targets, metrics, chronology and checkpoint records. No new JEV inference, GPU or Runpod operation was used.

## Other reported CPU results

The [real-pattern detector-agreement table](../../06b-chart-patterns/results/detector_agreement.csv) remains the reported real-data result; all 150 real parents are showcase-only, with no aggregate or per-class model accuracy claims. [Candle evaluation](../../06a-candles/results/eval_v7) and [synthetic evaluation](../../06b-chart-patterns/results/eval_v7) include CPU baselines, confusion/reliability/risk-coverage plots and paired arm comparisons. An independent CPU recalculation matched accuracy, macro-F1, Brier and confusion matrices in all 24 saved evaluation reports. The [runbook](../../GPU_RUNBOOK.md) retains the full 9,820-call schedule and unchanged server/runtime requirements.
