# Experiment 07: candle shapes and chart formations together for direction

**Real inference and v1 evaluation are complete.** AutoTrust's [JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL) forecasts the next five trading sessions with candle-shape and chart-formation directional guidance. The weights remain frozen; this tests prompting, not fine-tuning.

## V1 results: no added forecasting benefit

Real predictions were recorded in [4106a8e](https://github.com/mcravi8/mcravi8-lab/commit/4106a8e), followed by evaluation in [2419ec4](https://github.com/mcravi8/mcravi8-lab/commit/2419ec4). The cohort has 300 observations across 50 forecast dates and 49 tickers. Its descriptive test-majority base rate is **36.67%** (Up: 110/300). The train-frequency prior uses 33,041 outcomes completed by 2018-12-31, without fitting to these test labels.

| Arm / baseline | N | Accuracy | Brier |
| --- | ---: | ---: | ---: |
| Zero-shot | 300 | 37.33% | 0.862890 |
| Candle guidance | 300 | 38.00% | 0.830781 |
| Formation guidance | 300 | 36.67% | 0.875197 |
| Combined guidance | 300 | 39.00% | 0.859144 |
| Train-frequency prior | 300 | 36.67% | 0.665765 |
| Uniform probabilities | 300 | 36.67% | 0.666667 |
| Combined, image-control subset | 100 | 39.00% | 0.837441 |
| Combined, blank control | 100 | 27.00% | 1.105721 |

Lower Brier is better. The final two rows use the same 100 observations; compare those rows, not blank control against the full cohort. Uniform accuracy reflects the fixed Up-first argmax tie rule, not random guessing.

| Paired comparison | Brier reduction | 95% block-bootstrap CI |
| --- | ---: | ---: |
| Combined vs zero-shot (primary) | +0.003746 | [-0.019406, +0.028114] |
| Combined vs candles | -0.028364 | [-0.040751, -0.016581] |
| Combined vs formations | +0.016052 | [+0.005587, +0.027316] |
| Combined vs train-frequency prior | -0.193379 | [-0.252753, -0.134937] |
| Combined vs blank (100 paired charts) | +0.268280 | [+0.084149, +0.435918] |

Positive reduction favors combined guidance. Intervals use 2,000 paired moving-block resamples, seed 71, blocks of four consecutive retained forecast dates, with all tickers on each date kept together. The [full metrics](results/evaluation_v1/metrics.json) include 1/2/8-date sensitivities, calibration and repeatability. These exploratory intervals condition on fixed predictions and have no multiplicity correction or independent confirmation.

**Pattern guidance did not improve next-week direction under the frozen continuation criteria. No JEV arm beat the prior Brier; only the image check passed.** Combined guidance's primary interval includes zero, it trails candle-only guidance on Brier, and it is substantially worse than the prior. The image beats its paired blank control, but that does not establish forecasting skill. All five repeatability gates passed with 20/20 argmax matches per arm; this is separate from the performance checks.

This pairs with [Experiment 06](../06-jev-patterns/README.md#v7-results-useful-recognition-substantial-class-gaps): definitions improved pattern recognition, while its CPU bridge and this direct guidance pilot found no added five-session forecasting benefit. Recognition improves; forecasting does not in these tested designs. No trading costs or P&L were evaluated.

## Frozen comparison

The same 300 real observations enter four arms: zero-shot, candle guidance, formation guidance, and combined guidance. Each PNG has an independently scaled 100-session candle chart above a zoom of its final 20 sessions. Both panels end at the same bar. Axes, prices, dates, tickers and labels are absent; volume is gray. All four arms see identical pixels and the same causal volatility/target question. The combined arm concatenates the exact two single-arm guidance texts, with the same shared caution. A combined-guidance blank-image control uses 100 of the same observations, two per forecast date.

Candles cover Hammer, Shooting Star, Bullish/Bearish Engulfing, Morning/Evening Star and Doji. Formations cover H&S/inverse H&S, Double Top/Bottom, triangles and bull/bear flags. The instructions connect structure and preceding trend to conditional Up/Down hypotheses. Doji, absent patterns and disagreement do not automatically mean Flat. Only already visible confirmation can count; long-ago breakouts may already have exhausted their move. The conventional interpretations follow [StockCharts' bullish candle guidance](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/candlestick-charts/candlestick-bullish-reversal-patterns), [bearish guidance](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/candlestick-charts/candlestick-bearish-reversal-patterns), [chart-pattern descriptions](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-patterns), [H&S completion](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-patterns/head-and-shoulders-top) and [flag continuation](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-patterns/flag-pennant). These are hypotheses to test against actual returns, not evidence of a five-session edge.

[protocol.json](protocol.json) freezes the selection seed, prompts, target, comparison and call budget before preparation derives outcomes. Fifty forecast dates are chosen by seeded hash rank with fixed year quotas: six each in 2019–2025 and eight in 2026. Six eligible tickers per date are chosen from the verified cached Experiment 06 universe, without inspecting future prices, pattern labels or target classes. Holiday weeks end at their actual final observed session. All 100-bar input windows start on/after 2019-04-01; target history ends by the cached 2026-09-30 cutoff. This is a naturally selected pilot within the hand-selected universe, not a pattern-balanced dataset or evidence that both pattern types appear in every chart.

Return is `Close[t+5] / Close[t] - 1`. Flat means absolute return strictly below one-third of the sample standard deviation of 60 trailing overlapping five-session returns, including chart end (65 closes). Threshold equality is directional. The row's causal sigma and exact threshold are in the request; future return, truth, ticker and date are never sent. A constant prior uses only eligible weekly outcomes completed by 2018-12-31. Uniform is another probability baseline.

Primary comparison is **combined versus zero-shot paired Brier**. Report accuracy, macro-F1, confusion, log loss, Brier, ECE/reliability and confidence coverage. Four consecutive retained forecast-date blocks carry all tickers together; 1/2/8-block sensitivities are also reported. Fifty dates, rather than 300 rows, describe the principal common-shock clusters. Calendar gaps and overlapping input windows remain limitations; intervals condition on fixed predictions, have no multiplicity correction, and are exploratory. Single-guidance comparisons and combined-versus-blank are prespecified secondary comparisons. Continuation requires the primary four-block CI wholly above zero, combined Brier below the historical prior and both single-guidance point Briers, and positive four-block combined-versus-blank CI. Passing would justify independent validation, not a trading claim.

Historical data, a surviving universe and prompts chosen after Experiments 01–06 limit independence. Year quotas define the sampling distribution; the sample is not weighted to a live portfolio. Prompt lengths differ and this pilot does not isolate length effects or test other wordings. Unknown JEV pretraining overlap remains. This pilot does not modify Experiment 05's prospective holdout or Experiment 06's frozen results. No costs, execution or P&L are evaluated.

## Prepared inputs and validation

The frozen pilot has **300 observations across 50 dates and 49 tickers**, from 2019-08-30 through 2026-07-02. Target counts are **Up 110 / Flat 86 / Down 104**. All 300 five-session outcomes and trailing scales were independently recomputed with pandas; all visible-window starts and paired blank identities were checked. Nine CPU tests and GitHub CI pass. The actual-image fake smoke run completed ten predictions per arm plus all five repeatability gates: 250 synthetic calls, zero failures, no model loaded. These are setup checks, not model results. See [setup_validation.json](results/setup_validation.json) and [preparation.json](results/preparation.json). The source/protocol freeze was pushed as [5e387bd](https://github.com/mcravi8/mcravi8-lab/commit/5e387bd) before deriving this cohort's outcomes.

Review [sample 1](results/samples/sample_1.png), [sample 2](results/samples/sample_2.png), and [sample 3](results/samples/sample_3.png).

## CPU preparation and checks

Run from the repository root using the existing pinned Experiment 06 research environment:

```sh
06-jev-patterns/.venv/bin/python -m unittest discover -s 07-jev-pattern-direction/tests -v
06-jev-patterns/.venv/bin/python 07-jev-pattern-direction/src/prepare.py
06-jev-patterns/.venv/bin/python 07-jev-pattern-direction/scripts/run_gpu.py
```

Preparation only reads verified existing market snapshots; no data downloads occur. Chart PNGs are ignored under `charts/`. Preserve them for the GPU host; a fresh clone is insufficient. Three review examples are saved as `results/samples/sample_1.png` through `sample_3.png`. `results/preparation.json` records rendering dependencies and freezes every index, prior and image hash. Re-running preparation verifies the existing freeze and returns without rerendering. Changed assets or source require a separately documented version.

For a loopback fixture, start `06-jev-patterns/tests/fake_jev_server.py`, then:

```sh
06-jev-patterns/.venv/bin/python 07-jev-pattern-direction/src/score.py \
  --base-url http://127.0.0.1:8765 --dry-run --limit 10
```

Fake outputs go only to `results/dry_run/`. They test wiring, not forecast quality. Partial predictions cannot be evaluated as a complete result.

## Real inference on the prepared Runpod host

Use the same cached Hugging Face snapshot/revision and native System 1 server as Experiment 06, network volume `jev-weights`, US-MO-1, port 8000. Reuse [the pinned runtime and Linux scorer setup](../06-jev-patterns/GPU_RUNBOOK.md#observed-build-pins-and-linux-scorer): `/root/venv/bin/python3` for the server, `/root/svenv/bin/python` for scoring. Torch 2.13.0/torchvision 0.28.0 are the observed PyPI distributions; Torch uses CUDA 13.0 and requires driver >=580. The vLLM nightly remains the exact observed build. No package/model download, pod creation or lifecycle operation is performed by this script.

A local transfer archive is ready at `07-jev-pattern-direction/.cache/charts_v1.tar.gz` (2944985 bytes; SHA-256 `45cfd12b6b641072943cee2da7a33d9c5920619cdfd6c20c0a851a993840b98e`). It contains only the 300 ignored charts, with repository-relative paths. The indexes, prompts, prior, freeze and samples are tracked in Git. After pulling the latest repository on the pod and copying this archive to `/root/charts_v1.tar.gz`, restore it:

```sh
cd /root/repo
sha256sum /root/charts_v1.tar.gz
tar -xzf /root/charts_v1.tar.gz -C /root/repo
/root/svenv/bin/python 07-jev-pattern-direction/scripts/run_gpu.py
```

The final command verifies the bundle and prints the 1,500-call plan without starting a GPU server.

Transfer the repository plus all 300 ignored PNGs to the existing host. The script reuses Experiment 06's weight/runtime verification and native server flags: serial batch one, eager, no prefix cache or chunked prefill, no batch invariance for GDN_ATTN. It does not invoke plain vLLM chat or the older Desktop wrapper. The native serving source is checksum-verified against the preserved Experiment 01/06 contract.

```sh
cd /root/repo
/root/svenv/bin/python 07-jev-pattern-direction/scripts/run_gpu.py \
  --execute --model-dir /workspace/models/JEV-27B-VL \
  --server-python /root/venv/bin/python3 --scorer-python /root/svenv/bin/python
```

Every arm scores 20 charts twice first and aborts on the frozen v7 repeatability policy: mean absolute probability drift <=0.01; every argmax mismatch must have a top-two margin <=0.01 in either pass. Max/p90 drift, flagged charts, raw agreement and near-tie exceptions remain reported. That policy was changed post-hoc in Experiment 06; it is retained without further relaxation here and is not HF/vLLM equivalence or an accuracy guarantee. A failed arm prevents a complete paired result. Resume preserves completed rows but repeats a pending arm's live gate and rejects altered provenance.

V1 completed the **1,500-call plan**: 1,200 main predictions + 100 blank controls + 200 repeatability calls. The [session summary](results/session_v1/session_summary.json) records 1,095.468 seconds (18.26 minutes) elapsed; this is session time, not billed pod time. Scoring has a 60-minute timeout. The script stops its own server; it does not stop cloud pod billing.

To reproduce the v1 evaluation on CPU:

```sh
06-jev-patterns/.venv/bin/python 07-jev-pattern-direction/src/evaluate.py
```

Predictions/provenance/parity/error files go to `results/inference/v1/`; server/runtime evidence goes to `results/session_v1/`; paired metrics/report go to `results/evaluation_v1/`. Never infer scientific performance from the fake server.
