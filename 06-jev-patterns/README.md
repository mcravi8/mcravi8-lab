# Experiment 06: Can JEV-27B-VL recognize chart patterns?

Day 1 is CPU-only. Shared infrastructure is implemented; the candle and synthetic dataset stages follow in separate commits. No JEV inference, GPU training, or Runpod resource has been started. [PLAN.md](PLAN.md) preserves the original design.

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

The renderer reuses Experiment 01's white background, colored candle/volume panels and absence of text, axes or identifying PNG metadata. Its interface is `render_chart(frame, path, n_bars=20, size=(640,480), style="candle", mirror=False)`. It accepts exactly the supplied visible bars; scales never depend on future data. `style` also accepts `ohlc` or `line`; a mirror reverses pixels horizontally.

All `data/`, `charts/`, `.cache/`, `.venv/` and model/checkpoint directories are Git-ignored at every nesting level. Selected result images and provenance are committed.

## Scoring contract and parity gate

[jev_api.py](shared/src/jev_api.py) generalizes Experiment 03's native **`POST /v1/decide`** client. Each call sends a base64 PNG, a question, an ordered list of 2–256 options, and `thinking="off"`. An optional definitions file prefixes the question. The documented `single` strategy is explicit; up to 16 choices use the native readout, while larger lists use the server's single-strategy adaptation. The actual tasks have eight and seven choices.

No labels, ticker names, session dates, filenames or split fields are sent to the model. Responses must contain one valid probability per option; reordered options are mapped back. Protocol, served model name, calibration and model revision are checked. The model revision must be advertised by the server or supplied as an immutable 40-character Hugging Face commit / 64-character weights hash. A caller-declared revision is recorded as such, not treated as server attestation. `--served-model-name jev` supports that explicit serving alias while retaining the canonical model ID.

Before any pending predictions, **20 deterministically selected charts are scored twice**. Maximum absolute probability difference must be **<= 0.01**. Failure blocks scoring and is logged. Resumed invocations with pending rows repeat this gate because the same URL and weights can hide a restarted runtime. `--parity-only` always performs the 40 calls. A completed no-op resume makes no inference calls.

This gate checks repeatability. It does **not** prove Hugging Face/vLLM implementation parity or model accuracy; the separate Experiment 04 HF-vLLM gate is still required before its training workflow.

Scoring writes index metadata plus `probabilities` (JSON array in option order), `prediction`, and `latency_ms`. Adjacent `.meta.json`, `.parity.json`, `.summary.json`, and `_errors.csv` files record provenance, gate results, progress and failures. Resume skips valid predictions, repairs an interrupted final CSV record, refuses mismatched input/model/context provenance, and locks an output against concurrent writers. Errors return a nonzero exit status and can be retried by rerunning the same command. Each arm needs its own output path.

Supported CLI/environment configuration: `--base-url` / `JEV_BASE_URL`, `--served-model-name` / `JEV_SERVED_MODEL`, `--model-revision` / `JEV_MODEL_REVISION`, `--batch-size` / `JEV_BATCH_SIZE` (16), `--concurrency` / `JEV_CONCURRENCY` (4), `--timeout` / `JEV_TIMEOUT` (60 seconds), `--retries` / `JEV_RETRIES` (3). Optional authentication uses `JEV_API_KEY`. Remote endpoints must use HTTPS. Controls are `--control image|mirror|blank`.

For local contract validation, start `tests/fake_jev_server.py` in another terminal. `--dry-run` only permits loopback and marks its outputs synthetic; fake results are not model results. The test suite exercises 20-chart parity, ten predictions, partial resume, complete no-op resume, drift, retries, provenance mismatch, aliases, more than 16 options and damaged CSV recovery.

## Evaluation and CPU baselines

`shared/src/evaluate.py` reports accuracy, macro-F1 over every declared class, a confusion matrix, multiclass Brier (sum over classes), log loss, 15-bin top-confidence ECE, reliability plots, and error risk/coverage at confidence 0.5/0.7/0.9. Empty accepted sets have null risk, not zero. All argmax ties use the first ordered option.

Real-data intervals resample whole **ticker-calendar-quarter** blocks. Synthetic intervals group all noise/framing versions by `template_id`. Missing or insufficient blocks yield explicit unavailable intervals. Paired comparisons align exact file IDs and truth, use the same bootstrap draw for both arms, and report exact two-sided McNemar. Its independence assumption is stronger than the block bootstrap's: correlated discordant rows invalidate its nominal p-value. Ticker-quarter intervals also omit cross-ticker common shocks and dependence lasting longer than a quarter. These are exploratory intervals, with no multiplicity correction.

`shared/src/baselines.py` supplies training-majority, training-frequency prior, uniform probabilities, numerical logistic regression and the Experiment 04 CNN architecture with an arbitrary output count. Logistic uses only normalized visible OHLCV interpolated to 32 positions, with its scaler fitted on training rows. CNN validation comes from the latest 20% of training dates with overlapping fit windows purged. It verifies checkpoint reload predictions. Synthetic examples cannot train fitted baselines; uniform needs no training data. Default CLI mode is a preflight; fitting needs explicit `--mode train`.

All real training dates end by **2018-12-31**. January–March 2019 is embargoed, and test input windows begin on or after **2019-04-01**. Synthetic data are evaluation-only. The train set is for baseline fitting and any future calibration; held-out labels must not select prompts, thresholds, hyperparameters or examples.

## Dataset stages and Day 2

- [06a candles](06a-candles/README.md): TA-Lib plus independent pandas labels, exact agreement, exclusive labels, ticker/year-stratified selection, twenty-candle images and a fixed visual audit.
- [06b synthetic](06b-chart-patterns/README.md): 200 clean templates and five paired noise levels (1,000 noisy images), seven classes, sixty to 120 bars, 1024 x 640 images.

Day 2 adds real multi-bar pattern labels: verify the external detector's supported classes and license, implement independent zigzag rules, require agreement and a completed pattern, apply exclusive-label and date/window checks, target 80–120 examples per class, and hand-review 10–15 per class. Freeze the final inference-arm manifest and two prompt paraphrases per arm before scoring. Prepare the fixed 100-bar framing control and the 200-chart style subset.

One geometric correction to the preserved plan: a **horizontal mirror reverses time**; it does not exchange Head and Shoulders with Inverse or Double Top with Double Bottom. Price inversion creates those counterparts. Horizontal controls therefore have no assumed class swap. The current shared mirror is horizontal.

Real JEV scoring, baseline fitting on the real dataset, calibration, arm comparisons and any Runpod/GPU session remain pending. Day 1 software tests and detector agreement do not establish visual-recognition accuracy or trading value.
