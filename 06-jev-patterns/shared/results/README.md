# Experiment 06 results and repeatability noise floor

No v7 GPU inference has run. This report analyzes user-supplied candle gates on CPU. [V6 original failures and SHA-256s](diagnostics/2026-10-11-v6-parity/diagnosis.json), [v5 evidence](diagnostics/2026-10-10-v5-parity/diagnosis.json), and [recomputed statistics](parity_noise_floor.json) are preserved.

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

These measurements concern one fixed 20-chart candle fixture and are not independent validation, a general noise bound, pattern-task parity, HF-vLLM equivalence or recognition accuracy. Later results must disclose the adaptations, residual noise floor and allowed mismatch counts. Small probability/calibration changes need interpretation against the observed variation. Historical pair replay satisfies v7, but every future pending arm still requires fresh live gates.

## Other reported CPU results

The [real-pattern detector-agreement table](../../06b-chart-patterns/results/detector_agreement.csv) remains the reported CPU result; all 150 real parents are showcase-only, with no aggregate or per-class model accuracy claims. The [runbook](../../GPU_RUNBOOK.md) retains the 9,820-call schedule and unchanged server/runtime requirements.
