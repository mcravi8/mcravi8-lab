# Experiment 06: Can JEV-27B-VL recognize chart patterns?

**Days 1–2 CPU preparation is complete:** 1,200 candle charts, 1,200 synthetic charts, 150 real pattern parents (300 paired images), framing/style/placebo controls, frozen arms and a future GPU runbook. All real positive classes miss the target. No JEV inference, GPU training, or Runpod resource has been started. [PLAN.md](PLAN.md) preserves the original design.

This experiment studies recognition of visible patterns, rather than future returns. It uses **AutoTrust's [JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL)**, with model weights, calibration and native System 1 serving code from Hugging Face. A later authorized session may serve the model on Runpod. Local validation uses a fake server and CPU tensors only.

## Separate pinned environment

From the repository root, use Python 3.11. TA-Lib comes from a pip wheel, including its C library; no Homebrew installation is needed.

```sh
python3.11 -m venv 06-jev-patterns/.venv
06-jev-patterns/.venv/bin/python -m pip install --only-binary=TA-Lib -r 06-jev-patterns/requirements.txt
# macOS CPU baseline/checkpoint tests:
06-jev-patterns/.venv/bin/python -m pip install -r 06-jev-patterns/requirements-cnn.txt
# On Linux, install the CPU-only Torch wheel instead:
# 06-jev-patterns/.venv/bin/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
06-jev-patterns/.venv/bin/python -m unittest discover -s 06-jev-patterns/tests -v
```

Direct dependency versions follow Experiment 05 and add `TA-Lib==0.8.1`. CNN code uses `torch==2.14.1` and explicitly selects CPU. The downloaded TA-Lib wheel reports C library 0.8.1. CI installs wheels and runs all historical suites plus Experiment 06 without downloading model weights or market data.

## Data and rendering

The reviewed partial downloader and [50-name universe](shared/config/universe.json) are retained and hardened. The universe includes all 12 Experiment 03 tickers. Adjusted daily Yahoo OHLCV spans **2005-01-01 through 2026-09-30**, starting at IPO where later. It is a present-day, hand-selected universe with survivorship bias.

```sh
06-jev-patterns/.venv/bin/python 06-jev-patterns/shared/src/data.py
```

Validation rejects missing/nonfinite values, duplicate/unordered sessions, negative volume and invalid OHLC bars. It never fills, drops or clips bars. Tiny adjusted-price rounding differences use a relative `1e-8` tolerance. Existing snapshots resume only after validation and checksum verification; `--refresh` explicitly replaces them. [data_manifest.json](shared/results/data_manifest.json) records all 50 inputs, configuration and hashes. Yahoo can revise adjusted history: preserve the ignored local snapshots for exact reproduction.

The renderer reuses Experiment 01's white background, colored candle/volume panels and absence of text, axes or identifying PNG metadata. Its interface is `render_chart(frame, path, n_bars=20, size=(640,480), style="candle", mirror=False, volume_color="direction")`. It accepts exactly the supplied visible bars; scales never depend on future data. `style` also accepts `ohlc` or `line`; a mirror reverses pixels horizontally. Synthetic charts select `volume_color="neutral"`, making every volume bar gray independently of candle direction. Volume values use the same independent distribution across source classes.

All `data/`, `charts/`, `.cache/`, `.venv/` and model/checkpoint directories are Git-ignored at every nesting level. Selected result images and provenance are committed.

## Scoring contract and parity gate

[jev_api.py](shared/src/jev_api.py) generalizes Experiment 03's native **`POST /v1/decide`** client. Each call sends a base64 PNG, a question, an ordered list of 2–256 options, and `thinking="off"`. An optional definitions file prefixes the question. The documented `single` strategy is explicit; up to 16 choices use the native readout, while larger lists use the server's single-strategy adaptation. The actual tasks have eight and seven choices.

No labels, ticker names, session dates, filenames or split fields are sent to the model. Responses must contain one valid probability per option; reordered options are mapped back. Protocol, served model name, calibration and model revision are checked. The model revision must be advertised by the server or supplied as an immutable 40-character Hugging Face commit / 64-character weights hash. A caller-declared revision is recorded as such, not treated as server attestation. `--served-model-name jev` supports that explicit serving alias while retaining the canonical model ID.

Before any pending predictions, **20 deterministically selected charts are scored twice**. Every top prediction must match between passes **AND maximum absolute probability difference must be <=0.03**. Ties use the first declared option. Matteo approved this v5 rule **after observing failures**, replacing the original 0.01-only rule; see the [policy history](GPU_RUNBOOK.md#failed-gate-diagnosis-and-decision). Failure blocks scoring and is logged. Resumed invocations with pending rows repeat this gate because the same URL and weights can hide a restarted runtime. `--parity-only` always performs the 40 calls. A completed no-op resume makes no inference calls.

This gate checks repeatability. It does **not** prove Hugging Face/vLLM implementation parity or model accuracy; the separate Experiment 04 HF-vLLM gate is still required before its training workflow.

Scoring writes index metadata plus `probabilities` (JSON array in option order), `prediction`, and `latency_ms`. Adjacent `.meta.json`, `.parity.json`, `.summary.json`, and `_errors.csv` files record provenance, gate results, progress and failures. Resume skips valid predictions, repairs an interrupted final CSV record, refuses mismatched input/model/context provenance, and locks an output against concurrent writers. Errors return a nonzero exit status and can be retried by rerunning the same command. Each arm needs its own output path.

Supported CLI/environment configuration: `--base-url` / `JEV_BASE_URL`, `--served-model-name` / `JEV_SERVED_MODEL`, `--model-revision` / `JEV_MODEL_REVISION`, `--batch-size` / `JEV_BATCH_SIZE` (16), `--concurrency` / `JEV_CONCURRENCY` (4), `--timeout` / `JEV_TIMEOUT` (60 seconds), `--retries` / `JEV_RETRIES` (3). Optional authentication uses `JEV_API_KEY`. Remote endpoints must use HTTPS. Controls are `--control image|mirror|blank`.

For local contract validation, start `tests/fake_jev_server.py` in another terminal. `--dry-run` only permits loopback and marks its outputs synthetic; fake results are not model results. The test suite exercises 20-chart parity, ten predictions, partial resume, complete no-op resume, drift, retries, provenance mismatch, aliases, more than 16 options and damaged CSV recovery.

## Evaluation and CPU baselines

`shared/src/evaluate.py` reports accuracy, macro-F1 over every declared class, a confusion matrix, multiclass Brier (sum over classes), log loss, 15-bin top-confidence ECE, reliability plots, and error risk/coverage at confidence 0.5/0.7/0.9. Empty accepted sets have null risk, not zero. All argmax ties use the first ordered option.

Real-data intervals resample whole **ticker-calendar-quarter** blocks. Synthetic intervals group all noise/framing versions by `template_id`. Missing or insufficient blocks yield explicit unavailable intervals. Paired comparisons align exact file IDs and truth, use the same bootstrap draw for both arms, and report exact two-sided McNemar. Its independence assumption is stronger than the block bootstrap's: correlated discordant rows invalidate its nominal p-value. Ticker-quarter intervals also omit cross-ticker common shocks and dependence lasting longer than a quarter. These are exploratory intervals, with no multiplicity correction.

`shared/src/baselines.py` supplies training-majority, training-frequency prior, uniform probabilities, numerical logistic regression and the Experiment 04 CNN architecture with an arbitrary output count. Logistic uses only normalized visible OHLCV interpolated to 32 positions, with its scaler fitted on training rows. CNN validation comes from the latest 20% of training dates with overlapping fit windows purged. It verifies checkpoint reload predictions. Synthetic examples cannot train fitted baselines; uniform needs no training data. Default CLI mode is a preflight; fitting needs explicit `--mode train`.

All real training dates end by **2018-12-31**. January–March 2019 is embargoed, and test input windows begin on or after **2019-04-01**. Synthetic data are evaluation-only. The train set is for baseline fitting and any future calibration; held-out labels must not select prompts, thresholds, hyperparameters or examples.

## 06a dataset ready

The candle dataset has **1,200 charts: 150 per class, each with 75 train and 75 test**. Doji, Hammer, Shooting Star, Bullish/Bearish Engulfing, Morning/Evening Star and None all meet the quota. TA-Lib and the independent pandas implementation agree on all 254,240 eligible windows; per-class positive agreement is 100%. The two detectors implement the same TA-Lib rulebook: 100% agreement is a software consistency check, **not a label-quality check**. The rarest single-label pools are Evening Star (560) and Morning Star (590). The observed label-clarity measure is the fixed visual audit: **10/50 subtle or ambiguous (20%)**, with 40/50 visually consistent (80%). All cases remain in the dataset. This AI-assisted, stratified audit measures observed visual clarity in the reviewed sample; it is not an independent expert error-rate estimate for all charts.

Each index row now includes a frozen `clear` flag based on body/wick, gap, penetration and trend margins well inside the class thresholds. None is clear only when it also avoids expanded near-pattern geometries. [clear_rules.json](06a-candles/clear_rules.json) records the margins; [clarity_summary.json](06a-candles/results/clarity_summary.json) reports class/split coverage. The margins select **371/1,200 charts (193 train, 178 test)**. These heuristic flags support secondary analysis and do not replace the empirical visual audit or change the 1,200 primary labels. `score.py` and baseline outputs retain the flag; `evaluate.py --clear-only` restricts both paired arms to their flagged rows and rejects mismatched coverage. Report full-set results alongside the clear subset.

Nine 640 x 480 samples are in [06a-candles/results/samples](06a-candles/results/samples); the sample manifest maps filenames to source labels and dates. See the [candle README](06a-candles/README.md) for all counts, exclusions and reproduction commands.

Future scoring commands below require an already authorized, separately verified native JEV server. They do not provision a pod. Use its exact Hugging Face snapshot commit in `JEV_MODEL_REVISION` and its documented served alias if necessary.

```sh
# Optional pure CPU preflight; does not train:
06-jev-patterns/.venv/bin/python 06-jev-patterns/shared/src/baselines.py \
  --root 06-jev-patterns/06a-candles --index 06-jev-patterns/06a-candles/results/index.csv \
  --options-file 06-jev-patterns/06a-candles/options.json \
  --out 06-jev-patterns/06a-candles/results/baselines

# Future zero-shot arm (JEV_BASE_URL and JEV_MODEL_REVISION must be set):
06-jev-patterns/.venv/bin/python 06-jev-patterns/shared/src/score.py \
  --root 06-jev-patterns/06a-candles --index results/index.csv \
  --options-file 06-jev-patterns/06a-candles/options.json \
  --question 'Which candlestick pattern ends at the right edge of this chart?' \
  --out results/zero_shot/preds.csv
```

For a definitions arm, add `--context-file 06-jev-patterns/shared/context/candles.txt` and select a new output such as `results/definitions/preds.csv`. Add `--parity-only` to perform only the gate. A local smoke run instead uses `--base-url http://127.0.0.1:8765 --dry-run --limit 10 --out results/dry_run/preds.csv`, with the fake server running. The examples illustrate the client; use the [frozen Day 2 config](config/inference_arms.json) and [single GPU runbook](GPU_RUNBOOK.md) for the prospective experiment.

```sh
# Later evaluation and exact paired comparison, using completed compatible arms:
06-jev-patterns/.venv/bin/python 06-jev-patterns/shared/src/evaluate.py \
  --predictions 06-jev-patterns/06a-candles/results/definitions/preds.csv \
  --compare 06-jev-patterns/06a-candles/results/zero_shot/preds.csv \
  --out 06-jev-patterns/06a-candles/results/comparison
```

## 06b synthetic dataset ready

There are **200 clean + 1,000 noisy charts** across seven source classes. Head and Shoulders, Inverse Head and Shoulders, Double Top and Double Bottom each have 29 clean + 145 noisy examples (174). Triangle, Bull Flag and None each have 28 clean + 140 noisy examples (168). Each clean template shares its geometry and noise realization with five increasing noise amplitudes: 0.035, 0.08, 0.16, 0.30 and 0.55 of its clean log-price range.

Protocol v3 replaces the straight-line None shortcut with screened random walks, choppy ranges, trends with random pullbacks, incomplete formations and structurally broken reversals/consolidations. The 28 clean None parents cover these six families with counts 5/5/5/5/4/4. Every parent independently varies pivot proportions and timing, padding/position, price scale, bar count, microtexture, opening gaps and wicks. Lead-in and lead-out directions are independent of the pattern direction. All classes use the same nuisance distributions, with each clean parent scaled to an independently drawn daily log-return volatility. Volume has an independent process and neutral gray bars.

The stronger nested noise affects major pivots as well as intervening bars; the hardest level can obscure the source formation and its breakout. Noisy labels identify the source construction. The conservative None screen operates on synthetic control points; incidental local formations can still occur. See the [protocol](06b-chart-patterns/protocol.json) for the exact geometric and stochastic definitions.

All images are 1024 x 640 with 60–120 bars. The [sample grid](06b-chart-patterns/results/samples/synthetic_grid.png) compares seven classes across clean, intermediate and strongest noise; the [diversity grid](06b-chart-patterns/results/samples/clean_diversity_grid.png) shows independent clean parents and all six None families. Labels identify the source construction; severe noise may introduce incidental shapes. The 200 template IDs, rather than 1,200 correlated variants, define the independent sampling units. No synthetic class has a deficit.

The [synthetic README](06b-chart-patterns/README.md) documents reproduction, the prepared fixed-100-bar control and geometric constraints. A future scoring call uses `--root 06-jev-patterns/06b-chart-patterns --index results/synthetic_index.csv --options-file 06-jev-patterns/06b-chart-patterns/options.json`; the definitions arm uses `shared/context/chart_patterns.txt` from this experiment. Evaluation automatically recognizes complete template IDs with blank market dates and uses template blocks. The 200 primary synthetic templates remain evaluation-only. Two separate, independently seeded cohorts are fitted solely for the CPU shortcut regression audit. The fixed logistic and shallow-forest checks score 63.44%/86.35% balanced accuracy on the regression cohort and 67.50%/88.13% on a predeclared fresh confirmation cohort, each below the unchanged 90% gate. The original generator scored 100%; an intermediate revision failed the forest gate at 95.42% and informed the final negative families. The [audit report](06b-chart-patterns/results/leakage_audit.json) and [preserved development history](06b-chart-patterns/results/leakage_development_history.json) document this adaptation. These checks do not prove all shortcuts absent; see the synthetic README for ordinary accuracy, class imbalance and remaining separation.

## Verification and Day 2 outcome

All **121 Experiment 06 tests and 60 historical tests passed locally** in the pinned environment. `pip check` passed. The Day 1 images and both sample collections were validated and visually inspected. The trimmed freeze checks all 4,400 unique target PNGs, their matching numeric bars and all 50 raw input hashes; see [day2_validation.json](shared/results/day2_validation.json). All seven real review grids and selected derived styles were inspected during Day 2; existing reference boards are inactive historical artifacts. The data cache verifier confirmed all 50 inputs without network access and preserved the frozen input-manifest hash. [GitHub CPU CI](https://github.com/mcravi8/mcravi8-lab/actions/workflows/tests.yml) runs on each staged push.

The candle definitions contain 157 words; chart-pattern definitions contain 255 words. They describe the frozen detector/generator rules. No real-model prediction, real-dataset baseline fit, GPU or Runpod call was made for Day 1.


- [06a candles](06a-candles/README.md): TA-Lib plus independent pandas labels, exact agreement, exclusive labels, ticker/year-stratified selection, twenty-candle images and a fixed visual audit.
- [06b synthetic](06b-chart-patterns/README.md): 200 clean templates and five paired noise levels (1,000 noisy images), seven classes, sixty to 120 bars, 1024 x 640 images.

Day 2 real detection is complete across all 50 input snapshots: **150 showcase-only parents (82 train / 68 test)**, with 6 H&S, 2 inverse, 1 Double Top, 1 Double Bottom, 4 Triangle, 36 Bull Flag and 100 None. All positive classes are rare; inverse and both doubles have no test examples. See the [real protocol, license, agreement rates and review grids](06b-chart-patterns/README.md#day-2-real-market-dataset). Fixed-100 historical frames are paired with variable frames, and user visual review remains pending. The [frozen arms](config/inference_arms.json) and [future GPU runbook](GPU_RUNBOOK.md) are now prepared. All 4,400 active target PNGs are hash-pinned with indexes and code. All 150 real parents remain in both main arms and framings, marked `showcase_only` / `accuracy_valid=false`; no aggregate or per-class model accuracy claims are permitted. The evaluator rejects their accuracy metrics. The [reported detector-agreement table](06b-chart-patterns/results/detector_agreement.csv) is a CPU detector-overlap result, not model accuracy or expert label validity.

One geometric correction to the preserved plan: a **horizontal mirror reverses time**; it does not exchange Head and Shoulders with Inverse or Double Top with Double Bottom. Price inversion creates those counterparts. Horizontal controls therefore have no assumed class swap. The current shared mirror is horizontal.

The trimmed frozen schedule has **21 jobs, 9,820 decide calls**: 8,900 predictions plus 920 parity calls. Corrected serial execution has a planning estimate of **103–310 GPU minutes** (103.2–309.6 before rounding). These are planning assumptions, not measured throughput; retries/resumes add calls. See the [budget and future command](GPU_RUNBOOK.md).

The **reference-image arm is removed**. Main zero-shot/definitions arms use one primary wording; the second wording runs only on **100 charts per sub-part**, for both arms. Style and horizontal-mirror subsets also have 100 charts each: test-only market candles for 06a and distinct clean synthetic parents for 06b. OHLC/line and mirror controls are zero-shot only, and candle style reuses primary predictions. An additional 100-parent eligible synthetic subset tests price-inversion class flips. Secondary subsets are fixed before inference and do not measure noisy-chart robustness. Variable/fixed-100 main collections remain paired: genuine historical bars for real showcases, same-parent interpolation for synthetic charts.

JEV scoring, benchmark baselines, calibration, arm comparisons and full benchmark inference remain pending. The first user-run GPU attempt failed its global candle gate before any predictions; real 06b charts remain descriptive showcases only. The corrected runbook has **not been executed** during this revision; CPU tests import helpers and mock execution. The supplied earlier GPU failure is preserved in [diagnostics](shared/results/diagnostics/2026-10-10/diagnosis.json). Neither tests nor detector agreement establish visual-recognition accuracy or trading value.


## Parity correction after the first GPU attempt

The user-run candle gate produced 3/20 probability differences over 0.01 (maximum 0.033344), with 20/20 unchanged argmaxes and no HTTP errors. The gate was already serial: prediction concurrency four did not run. Prefix-cache cold/warm paths and BF16/LoRA kernel reductions are plausible causes; a cu129 vLLM wheel paired with Torch CUDA13.0 remains a build risk, without a matched-build comparison.

After v4 drift **0.0184107** with **20/20 matching top predictions**, Matteo approved **all argmaxes matching AND max drift <=0.03** on 2026-10-10. This threshold changed after observing failures, with residual BF16 kernel nondeterminism under serial/eager/no-cache execution as the working explanation. We retain concurrency/batch one, eager mode, disabled prefix caching/chunking/async scheduling and seed zero. The subsequent v3 startup rejected **batch invariance for GDN_ATTN** before any decide calls; the flag is now removed from the policy and cleared from inherited server environments. The observed nightly **0.31.1rc1.dev253+g7d0b4e57a.cu129**, Torch distribution **2.13.0** / compiled CUDA **13.0**, torchvision **0.28.0**, and critical numerical dependencies are pinned and checked before launch. Torch/torchvision are installed from **PyPI**, with separate explicit requirements. **Runpod NVIDIA driver >=580** is required for CUDA13.0 and checked before Torch import/server startup; driver570 is rejected. Linux scoring uses an explicitly supplied on-pod interpreter such as **/root/svenv/bin/python**, with its own pinned requirements. Cached upstream serving files remain unchanged; the corrected launcher invokes the native server with explicit overrides.

The call plan stays **9,820**, with fresh frozen_v5 outputs. The serial runtime estimate is now **103–310 minutes**, unmeasured for the correction. See the [diagnosis](shared/results/diagnostics/2026-10-10/diagnosis.json), [GPU runtime contract](config/gpu_runtime.json) and [corrected runbook](GPU_RUNBOOK.md). The [v3 startup evidence](shared/results/diagnostics/2026-10-10-gdn-attn/diagnosis.json) (no measured drift) and [v4 gate evidence](shared/results/diagnostics/2026-10-10-v4-parity/diagnosis.json) (max drift 0.018410737935632693; 20/20 argmax agreement) are preserved with their original failure outcomes. All other execution/model/data settings remain frozen. No GPU was started for this revision; v5 still needs a real repeatability gate. Batch-invariant GDN_ATTN is unsupported in the pinned build.
