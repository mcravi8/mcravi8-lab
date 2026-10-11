# Experiment 06 frozen inference runbook

**Prepared and tested on CPU; never executed against JEV or a GPU.** The script [run_gpu.py](scripts/run_gpu.py) is the single future entry point. It does not create, start, stop or contact Runpod resources. Model inference requires a separately authorized, already prepared GPU host.

## Frozen assets and arms

[Model contract](config/model_contract.json) pins **autotrust/JEV-27B-VL** to Hugging Face snapshot **3514d1d8ead2189c75f433fa949fe63c8e3f3c66**, including all 19 base/adapter safetensor files, serving/calibration SHA-256s and tokenizer/config Git blob IDs. No model weights were downloaded for Day 2.

The [pinned model card](https://huggingface.co/autotrust/JEV-27B-VL/blob/3514d1d8ead2189c75f433fa949fe63c8e3f3c66/README.md), [native server](https://huggingface.co/autotrust/JEV-27B-VL/blob/3514d1d8ead2189c75f433fa949fe63c8e3f3c66/serve_decide.py) and [serve.sh](https://huggingface.co/autotrust/JEV-27B-VL/blob/3514d1d8ead2189c75f433fa949fe63c8e3f3c66/serve.sh) support a mixed text/image state and up to eight images. This keeps the **optional references arm**, using one labelled example board and one target image. Zero-shot and definitions calls use one target image. All calls use native System 1 `/v1/decide`, `thinking="off"`, `strategy="single"`, and ordered eight- or seven-class options.

[Inference config](config/inference_arms.json) freezes two distinct question paraphrases for each task, definitions/context hashes, reference boards, selected style IDs, input CSVs, image hashes, output paths and call budgets. No prompt is chosen by prediction performance.

| Input collection | Rows | Arms, each with two paraphrases |
|---|---:|---|
| 06a candles | 1,200 | zero-shot, definitions, optional definitions + references |
| 06b real variable / fixed historical 100 | 150 / 150 | same three arms |
| 06b synthetic variable / interpolated 100 | 1,200 / 1,200 | same three arms |
| 06a style OHLC / line | 200 / 200 | same three arms |
| 06b style OHLC / line | 200 / 200 | same three arms |
| Time-reversed 06a / real 06b / synthetic 06b | 1,200 / 150 / 1,200 | zero-shot placebo |
| Price-inverted eligible real + synthetic 06b | 14 + 864 | zero-shot class-equivariance control |

The candle style is an alias of each original row's prediction, avoiding duplicate calls. 06a's style subset has 200 distinct market parents; 06b uses **all 150 retained real parents plus 50 distinct clean synthetic parents** because fewer than 200 real parents qualify. Index fields `data_origin`, `source_collection`, `source_file` and parent IDs preserve pairing. The two origins must be analyzed separately. Training rows are scored descriptively; real primary evaluation uses **test only**. Synthetic evaluation clusters all noise, framing and style siblings by `template_id`.

Real fixed-100 charts use genuine historical observations with the same final date and contain the whole accepted pattern. Synthetic fixed-100 charts interpolate the same parent/noise realization; they are not independent examples. Numeric files match their corresponding framing or price transformation.

**Horizontal mirroring reverses time** and has no valid assumed class swap. Its config marks label accuracy invalid; use probability/position sensitivity only. A separate OHLC price inversion preserves chronology, maps H&S ↔ inverse and Top ↔ Bottom, and leaves Triangle unchanged. Bull Flag is excluded because it becomes an out-of-taxonomy Bear Flag; None is excluded because an unlabelled Bear Flag could become a Bull Flag. Volume stays unchanged and gray for 06b.

Reference boards are fixed before inference. Candles use the first clear training ID per class, never test rows. The eight reference rows' own training predictions are descriptive. Pattern references use seven independent synthetic seeds, distinct from the 200 evaluation parents; they are illustrations, not expert-validated real examples. [Reference provenance](shared/references/provenance.json) records identities/hashes. References are **off by default**, enabled only with `--include-references`.

## CPU preparation and transfer

The CPU dependencies remain pinned in the separate Experiment 06 environment. After restoring the exact ignored Day 1/Day 2 inputs:

```sh
06-jev-patterns/.venv/bin/python 06-jev-patterns/shared/src/prepare_day2.py
06-jev-patterns/.venv/bin/python -m unittest discover -s 06-jev-patterns/tests -v
```

Preparation writes fixed versions, styles, inversion charts, numeric bars, reference boards, arm indexes and the frozen config. [day2_preparation.json](shared/results/day2_preparation.json) records its config hash and counts. Regeneration is explicit: changed source, context or pixels changes the freeze.

Before a later authorized session, copy the repository **and all ignored chart PNGs referenced by the frozen config** to the GPU host, preserving repository-relative paths. A fresh Git clone alone is insufficient. Keep the market snapshots, scan caches and numeric bars for reproducibility even though scoring only reads PNGs. A changed/missing asset fails validation before server startup.

## Future GPU invocation — not run

The previously documented location is Runpod **US-MO-1**, network volume **jev-weights**, port **8000**. This runbook assumes such storage and compute are already prepared and authorized; it does not provision them. Cache the exact pinned Hugging Face snapshot and provide an installed, model-card-compatible vLLM environment.

Use the repository's corrected [Experiment 01 remote_serve.sh](../01-jev-tsla-charts/scripts/remote_serve.sh). The inspected `~/Desktop/runpod-setup/remote_serve.sh` was an older plain-vLLM launcher without the native decision route. It was neither changed nor executed. The repository wrapper invokes the snapshot's own `serve.sh`, retaining its LoRA, raw decision template and **`--max-num-seqs 8`** requirement. Scorer concurrency is four; batch size 16 only controls the client queue and does not override the server limit.

From the repository root on that future GPU host, with these example environment paths replaced by the actual prepared locations:

```sh
06-jev-patterns/.venv/bin/python 06-jev-patterns/scripts/run_gpu.py \
  --execute \
  --model-dir /workspace/models/JEV-27B-VL \
  --server-python /workspace/jev-venv/bin/python3 \
  --scorer-python 06-jev-patterns/.venv/bin/python
```

Append `--include-references` only to include the optional jobs. Without `--execute`, the script only validates local assets and prints the plan. **Neither mode was invoked during Day 2; tests import pure helpers and mock execution.**

The script:

1. Verifies all frozen input hashes and the cached model's weight/serving/config hashes; rejects an occupied port instead of replacing another server.
2. Records complete installed server/scorer package versions in `shared/results/inference_run/runtime_lock.json`; refuses a changed runtime on resume. The CPU environment is version-pinned. The model card does not supply a reproducible vLLM version lock; the future GPU environment must be prepared/validated, and GPU compatibility is currently unverified.
3. Starts only the verified native Experiment 01 wrapper, with offline Hugging Face mode, port 8000 and context length 16,384. No model downloads or package installs occur in the script.
4. Checks native identity/calibration, then runs **20 charts twice for each option schema** before any prediction job. Every pending job also runs its own 20-chart repeatability gate. Any max probability difference **>0.01 aborts** the session.
5. Scores every row of every selected frozen job, retaining index metadata, ordered probabilities, class prediction and latency. Timeout is 120 seconds with three retries. Resume rejects changed inputs, model, question, context or reference board and skips completed rows.
6. Writes each dataset's `results/inference/frozen_v1/<job>.csv` plus metadata, parity, error and summary files. Logs the server and overall session summary under `shared/results/inference_run/`. Stops its own server process on completion/failure. **It does not stop the existing cloud pod or its billing.**

Repeatability is not proof of HF/vLLM parity or recognition quality. Errors abort before later jobs; already completed outputs remain resumable. Global gates run again per session; partially completed jobs repeat their gate, while completed no-op jobs make no further decide calls.

## Call and time budget

| Plan | Jobs | Prediction calls | Job + global parity | Total decide calls | Estimated GPU minutes |
|---|---:|---:|---:|---:|---:|
| Default: zero-shot + definitions, controls | 44 | 25,656 | 1,760 + 80 | **27,496** | **120–313** |
| Include reference-image arm | 62 | 35,056 | 2,480 + 80 | **37,616** | **204–532** |

These are **fresh successful-run** counts, excluding retries and repeated partial resumes. Two paraphrases, fixed framing, styles and controls expand the original plan's rough 6,000–9,000-call estimate.

The time range assumes 0.25–0.65 effective GPU seconds per single-image call at concurrency four, weights reference calls twice, and adds 5–15 minutes for startup/weight validation. Experiment 03's historical median request latency was 594.704 ms; it is context, not measured throughput for these larger charts. No Experiment 06 GPU timing is available. Hardware, serial parity gates, image size and cache behavior can move runtime outside this planning range.

## Later evaluation

Use real test rows only, separate market and synthetic origins, and keep all correlated versions together. The evaluator accepts `--data-origin market --split test` or `--data-origin synthetic --split eval`; it rejects an unfiltered mixed-origin dataset. Add `--clear-only` for 06a's secondary analysis and report full-set results alongside it.

Time-placebo outputs are for sensitivity, not label accuracy. For style/framing comparisons, align parents using the frozen provenance rather than treating different PNG filenames as independent observations. The real 06b test set has only 18 positive examples and three missing classes; do not present it as a complete seven-class benchmark. Human review and any later dataset revision must be recorded separately from this initial freeze.
