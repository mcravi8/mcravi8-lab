# Experiment 06 trimmed inference runbook

**Prepared and tested on CPU; never executed against JEV or a GPU.** [run_gpu.py](scripts/run_gpu.py) is the single future entry point. It does not create, start, stop or contact Runpod resources. Inference requires a separately authorized, already prepared GPU host.

## Frozen assets and arms

The [model contract](config/model_contract.json) pins **autotrust/JEV-27B-VL** to Hugging Face snapshot **3514d1d8ead2189c75f433fa949fe63c8e3f3c66**, including all 19 base/adapter safetensor files, serving/calibration SHA-256s and tokenizer/config Git blob IDs. No model weights were downloaded for CPU preparation.

The [pinned model card](https://huggingface.co/autotrust/JEV-27B-VL/blob/3514d1d8ead2189c75f433fa949fe63c8e3f3c66/README.md) and [native server](https://huggingface.co/autotrust/JEV-27B-VL/blob/3514d1d8ead2189c75f433fa949fe63c8e3f3c66/serve_decide.py) define native System 1 `/v1/decide`, `thinking="off"`, `strategy="single"`, with ordered eight- or seven-class options. Every scheduled call has one target image. **The reference-image arm is removed**, including its enable flag; existing CPU reference boards are inactive historical artifacts.

[Inference config](config/inference_arms.json), schema 2, freezes **zero-shot and definitions** with one primary question wording per task. A second wording is confined to **100 charts per sub-part**, for both arms. These rows reuse main-arm predictions for the primary wording. No prompt or subset is selected from prediction performance.

| Input collection | Rows | Scheduled arms / wording |
|---|---:|---|
| 06a candles | 1,200 | zero-shot + definitions, primary wording |
| 06b real variable / genuine historical 100 | 150 / 150 | same arms/wording; **showcase-only** |
| 06b synthetic variable / interpolated 100 | 1,200 / 1,200 | same arms/wording |
| Paraphrase subset: 06a / synthetic 06b | 100 / 100 | both arms, second wording only |
| Style subset: OHLC and line, per sub-part | 100 each style | zero-shot, primary wording |
| Horizontal time mirror: 06a / synthetic 06b | 100 / 100 | zero-shot, primary wording |
| Eligible price inversion: synthetic 06b | 100 | zero-shot, primary wording |

Style and horizontal-mirror controls share the frozen 100-parent subsets: **06a market test rows**, and **06b distinct clean synthetic parents**, stratified by class. The paraphrase uses these same subsets. Candle style aliases original predictions and costs no new calls. Price inversion uses a separately fixed 100-parent clean synthetic subset from eligible classes. Real showcase charts are excluded from secondary controls. Main synthetic arms retain all 200 parents and six paired noise realizations; secondary controls do not measure noisy-chart robustness.

Index fields `evidence_role`, `accuracy_valid`, `data_origin`, `source_collection`, `source_file` and parent IDs preserve scope and pairing. All **150 real parents remain**, with both variable and genuine historical fixed-100 framing, in both main arms: 600 descriptive prediction calls. Config and CSVs mark these rows `showcase_only` and `accuracy_valid=false`. **No aggregate or per-class real-data model accuracy claims are allowed.** The evaluator rejects their metrics and paired accuracy comparisons. The reported CPU result is the [detector-agreement table](06b-chart-patterns/results/detector_agreement.csv), documented [with denominators and counts](06b-chart-patterns/README.md#reported-cpu-result-detector-agreement); detector overlap is not expert label quality or JEV accuracy.

Real fixed-100 versions use actual historical observations with the same final date and include the accepted pattern. Synthetic fixed-100 versions interpolate the same parent/noise realization; they are correlated examples. Numeric files match framing or price transformation.

**Horizontal mirroring reverses time**, with no valid assumed class swap; use probability/position sensitivity only. A separate OHLC price inversion preserves chronology and maps H&S ↔ inverse, Top ↔ Bottom, and Triangle → Triangle. Bull Flag is excluded because it becomes an out-of-taxonomy Bear Flag; None is excluded because an unlabelled Bear Flag could become a Bull Flag. Volume stays unchanged and gray for 06b.

## CPU preparation and transfer

CPU dependencies remain pinned in the separate Experiment 06 environment. Restore the exact ignored Day 1/Day 2 inputs, then run:

```sh
06-jev-patterns/.venv/bin/python 06-jev-patterns/shared/src/prepare_day2.py
06-jev-patterns/.venv/bin/python -m unittest discover -s 06-jev-patterns/tests -v
```

Preparation writes fixed versions, selected styles/inversions, numeric bars, arm indexes, the reported agreement CSV and frozen config. It removes obsolete generated arm indexes. [day2_preparation.json](shared/results/day2_preparation.json) records counts and the config hash. Changed source, context or pixels changes the freeze.

Before a later authorized session, copy the repository **and every ignored chart PNG referenced by the config** to the GPU host, preserving repository-relative paths. A fresh clone alone is insufficient. Keep snapshots, scan caches and numeric bars for reproducibility. Missing/changed assets fail validation before server startup.

## Future GPU invocation — not run

The documented Runpod location is **US-MO-1**, network volume **jev-weights**, port **8000**. Compute/storage must already be prepared and authorized; this script does not provision them. Cache the exact Hugging Face snapshot and provide an installed, model-card-compatible vLLM environment.

Use the corrected repository [Experiment 01 remote_serve.sh](../01-jev-tsla-charts/scripts/remote_serve.sh). The inspected `~/Desktop/runpod-setup/remote_serve.sh` was an older plain-vLLM launcher without the native decision route; it was neither changed nor executed. The repository wrapper invokes the snapshot's [serve.sh](https://huggingface.co/autotrust/JEV-27B-VL/blob/3514d1d8ead2189c75f433fa949fe63c8e3f3c66/serve.sh), preserving LoRA, the raw decision template and **`--max-num-seqs 8`**. Scorer concurrency is four; batch size 16 controls only the client queue.

From the repository root on that future GPU host, replace example paths with prepared environment locations:

```sh
06-jev-patterns/.venv/bin/python 06-jev-patterns/scripts/run_gpu.py \
  --execute \
  --model-dir /workspace/models/JEV-27B-VL \
  --server-python /workspace/jev-venv/bin/python3 \
  --scorer-python 06-jev-patterns/.venv/bin/python
```

There is no reference-image enable flag. Without `--execute`, the script validates local assets and prints the plan. **Neither mode was invoked during this CPU revision; tests import helpers and mock subprocess execution.**

The script:

1. Verifies frozen assets, the under-10,000 fresh call cap, showcase-only scope, and model weight/serving/config hashes; rejects an occupied port.
2. Records installed server/scorer versions in `shared/results/inference_run_v2/runtime_lock.json` and rejects changed runtimes on resume. CPU dependencies are pinned. The model card supplies no reproducible vLLM lock; future GPU compatibility must be validated and is currently unverified.
3. Starts the verified native Experiment 01 wrapper with offline Hugging Face mode, port 8000 and context length 16,384. No downloads or package installs occur.
4. Verifies native identity/calibration and runs **20 charts twice per option schema** before any prediction job. Every pending job also repeats 20 charts. Any maximum probability difference **>0.01 aborts**.
5. Scores all frozen jobs, preserving scope metadata, ordered probabilities, class predictions and latency. Timeout is 120 seconds with three retries. Resume rejects changed inputs, model, question or context and skips completed rows.
6. Writes `results/inference/frozen_v2/<job>.csv` plus metadata, parity, error and summary files. Logs and session summary go to `shared/results/inference_run_v2/`. It stops its own server on completion/failure, **not the existing cloud pod or its billing**.

Repeatability does not establish HF/vLLM parity or recognition quality. Global gates repeat each session; partial jobs repeat their gate. Completed no-op jobs make no new decide calls.

## Call and time budget

| Component | Jobs | Prediction calls |
|---|---:|---:|
| Main zero-shot + definitions, one wording | 10 | 7,800 |
| Second wording, 100 charts per sub-part and arm | 4 | 400 |
| OHLC / line, 100 per sub-part, zero-shot | 4 | 400 |
| Horizontal mirrors, 100 per sub-part, zero-shot | 2 | 200 |
| Price inversions, 100 synthetic parents, zero-shot | 1 | 100 |
| **Total predictions** | **21** | **8,900** |
| Per-job repeatability: 21 × 40 | | 840 |
| Two global option-schema gates: 2 × 40 | | 80 |
| **Total decide calls** | | **9,820** |

The fresh successful run is **9,820 calls**, down from 27,496, and remains below 10,000. Retries and repeated partial resumes are additional; this is a planned count, not a hard request limiter that would truncate output.

Estimated GPU time is **45.9–121.4 minutes (about 46–121)**: 0.25–0.65 effective GPU seconds per single-image call at concurrency four plus 5–15 minutes for startup/weight validation. Experiment 03's historical median request latency was 594.704 ms; it is context, not measured throughput for these charts. Serial parity gates, hardware, image size and cache behavior can move runtime outside the estimate. **No Experiment 06 GPU timing is available.**

## Later evaluation

Evaluate **06a test rows** (`--data-origin market --split test`) and **synthetic 06b** (`--data-origin synthetic --split eval`) separately. Retain all correlated versions within template blocks. The evaluator rejects mixed origins, showcase-only rows and accuracy-invalid time mirrors. `--clear-only` is secondary 06a analysis; report full-set results alongside it.

Real 06b is for visual examples and detector-agreement reporting only, regardless of train/test annotation. Do not report its per-class recognition accuracy. Align style/framing parents through frozen provenance; different PNG filenames are not independent observations. Horizontal controls report sensitivity rather than accuracy. User visual review remains pending; later dataset revisions must be separately recorded.
