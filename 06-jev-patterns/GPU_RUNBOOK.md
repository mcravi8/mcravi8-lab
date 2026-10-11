# Experiment 06 trimmed inference runbook

**Corrected configuration prepared/tested on CPU; not GPU verified.** A prior on-pod session failed the first global candle gate after 40 decide calls and wrote no predictions. [run_gpu.py](scripts/run_gpu.py) is the single future entry point. It does not create, start, stop or contact Runpod resources. Inference requires a separately authorized, already prepared GPU host.

## Frozen assets and arms

The [model contract](config/model_contract.json) pins **autotrust/JEV-27B-VL** to Hugging Face snapshot **3514d1d8ead2189c75f433fa949fe63c8e3f3c66**, including all 19 base/adapter safetensor files, serving/calibration SHA-256s and tokenizer/config Git blob IDs. No model weights were downloaded or GPU started during this correction.

The [pinned model card](https://huggingface.co/autotrust/JEV-27B-VL/blob/3514d1d8ead2189c75f433fa949fe63c8e3f3c66/README.md) and [native server](https://huggingface.co/autotrust/JEV-27B-VL/blob/3514d1d8ead2189c75f433fa949fe63c8e3f3c66/serve_decide.py) define native System 1 `/v1/decide`, `thinking="off"`, `strategy="single"`, with ordered eight- or seven-class options. Every scheduled call has one target image. **The reference-image arm is removed**, including its enable flag; existing CPU reference boards are inactive historical artifacts.

[Inference config](config/inference_arms.json), schema 3, freezes **zero-shot and definitions** with one primary question wording per task. A second wording is confined to **100 charts per sub-part**, for both arms. These rows reuse main-arm predictions for the primary wording. No prompt or subset is selected from prediction performance.

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

## Failed-gate diagnosis and decision

The [preserved diagnostics](shared/results/diagnostics/2026-10-10/diagnosis.json) include both passes, server log, original runtime inventory and session summary. The candle gate had **3/20 rows over 0.01**: 0.033344, 0.014644 and 0.013838. **All 20 argmax classes agreed**; all 40 requests succeeded. The session aborted before the pattern gate or predictions.

The scorer's parity function already sends one request at a time, completing the first 20 before the next 20. Its configured concurrency four applies only to later prediction jobs. The eight-choice native `single` route reads all option logprobs in one request; it does not submit eight parallel option requests. The mismatch between client concurrency four and server capacity eight therefore does **not** establish a batching cause for this failure.

Logs show BF16, seed zero, prefix caching with aligned hybrid/Mamba cache, chunked prefill, `torch.compile` and CUDA graphs, plus default LoRA kernel/JIT warnings. A cold first pass versus cached second pass can change the numerical work path; default LoRA reduction atomics are another plausible source. The observed vLLM wheel is tagged `cu129` while Torch was reported built for CUDA **13.0**. This build mixing is relevant, but constant across both passes: there is no coherent-build comparison or per-request cache trace. **Within-build kernel/cache-path numerical variation is the leading hypothesis; the diagnostics do not isolate its exact cause or clear the mixed build.**

**Decision (a): retain the predeclared 0.01 tolerance and change execution.** Moving to 0.05 after seeing this failure would weaken probability/Brier/calibration comparisons. Twenty stable argmaxes do not establish that larger probability drift is acceptable.

The corrected config uses:

- Scorer concurrency **1**, client batch size **1**, for both gates and predictions.
- `VLLM_BATCH_INVARIANT=1`. The [exact nightly revision's batch-invariance documentation](https://github.com/vllm-project/vllm/blob/7d0b4e57a/docs/features/batch_invariance.md) includes fixed-order LoRA shrink reduction, avoiding its default atomics.
- `--enforce-eager`, `--no-enable-prefix-caching`, `--no-enable-chunked-prefill`, `--no-async-scheduling` and `--seed 0`. The token budget is fixed at 16,384; GDN prefill is pinned to `triton`.
- Native LoRA `jev-decision`, processed logprobs, raw decision template, calibrated System 1 payload and **`--max-num-seqs 8`** retained. Changing the native read-out temperature to zero would change probability semantics, so it remains untouched.
- `PYTHONHASHSEED=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `NVIDIA_TF32_OVERRIDE=0`.
- Fresh `frozen_v3` outputs and `inference_run_v3` session files; preserve the failed v2 evidence.

[Upstream reproducibility guidance](https://docs.vllm.ai/en/latest/usage/reproducibility/) recommends batch invariance for online serving; reproducibility is conditional on the same hardware/version. This feature is beta, and JEV's Qwen3.5 hybrid/LoRA combination is not explicitly in the tested-model list. **These settings are an unverified deterministic correction, not a claimed successful GPU fix.** If the pinned build rejects a flag or still exceeds 0.01, abort and retain diagnostics. Do not silently disable the policy or increase tolerance.

## Observed build pins and Linux scorer

[GPU runtime contract](config/gpu_runtime.json) and [critical server requirements](requirements-jev-server.txt) pin the actually observed numerical stack:

| Component | Observed / required version |
|---|---|
| vLLM nightly | `0.31.1rc1.dev253+g7d0b4e57a.cu129` |
| Torch distribution | `2.13.0` |
| Torch compiled CUDA | **`13.0`**, reported by the user; checked through `torch.version.cuda` before launch |
| Triton | `3.7.1` |
| FlashInfer | `0.7.0.post1` |
| Transformers | `5.19.0` |
| NVIDIA CUDA runtime package | `13.0.96` |

The complete [original package inventory](shared/results/diagnostics/2026-10-10/runtime_lock.json) preserves the mixed CUDA 12/13 dependencies. CUDA 12.9 is **not** asserted as Torch's build. Package versions and imported build metadata are checked; a new nightly or CUDA build fails before serving. Original wheel URLs/hashes, container digest and GPU/driver identity were absent from supplied evidence, so these pins do not claim byte-identical reconstruction or binary compatibility. Reuse the original server environment/cached exact wheel; do not upgrade a moving nightly index. A coherent-build migration would be a separately recorded change.

A macOS virtual environment cannot be copied to Linux. Create or reuse the **on-pod Linux scorer** in a separate environment, matching the actual `/root/svenv` inventory through [requirements-scoring.txt](requirements-scoring.txt). The lightweight scorer needs no Torch/vLLM. From the prepared pod's repository root:

```sh
cd /root/repo
python3.11 -m venv /root/svenv
/root/svenv/bin/python -m pip install -r 06-jev-patterns/requirements-scoring.txt
/root/svenv/bin/python -m pip check
```

These are documented future-host setup commands; none were executed on a pod during this correction. The full research `06-jev-patterns/.venv` remains a local CPU preparation/testing environment.

## Future GPU invocation — correction not run

The Runpod location remains **US-MO-1**, network volume **jev-weights**, port **8000**. Compute/storage must already be authorized and prepared; this script does not provision them. The exact Hugging Face snapshot must already be cached.

The repository [Experiment 01 remote_serve.sh](../01-jev-tsla-charts/scripts/remote_serve.sh) and snapshot `serve.sh` remain unchanged and verified. Their hardcoded prefix-caching flag conflicts with this correction. The runbook therefore invokes the same checksum-verified **native `serve_decide.py` directly**, preserving the System 1 flags and applying the explicit overrides above. It does not use an ordinary vLLM chat launcher or modify cached upstream files.

Set `JEV_SERVER_PYTHON` to the actual Linux interpreter whose environment matches the observed server pins. The diagnostics did not preserve its exact path; the example below requires that operator-supplied path. Both interpreters must be explicit; there is no macOS venv fallback.

```sh
cd /root/repo
export JEV_SERVER_PYTHON=/absolute/path/to/observed/server/python
/root/svenv/bin/python 06-jev-patterns/scripts/run_gpu.py \
  --execute \
  --model-dir /workspace/models/JEV-27B-VL \
  --server-python "$JEV_SERVER_PYTHON" \
  --scorer-python /root/svenv/bin/python
```

There is no reference-image enable flag. Without `--execute`, the script validates local assets and prints the plan. **Neither mode was invoked during this CPU correction; tests import helpers and mock process launch.**

The script:

1. Verifies frozen assets, the under-10,000 call cap, showcase-only scope and model weight/serving/config hashes; rejects an occupied port.
2. Checks the exact required server/scorer package pins, `torch.version.cuda=13.0`, batch-invariance capability and supported CLI flags **before starting a server**. Runtime probes import packages/CLI help only; they do not create an inference engine.
3. Records packages, build/platform metadata and policy in `shared/results/inference_run_v3/runtime_lock.json`; rejects changed runtimes on resume. It records the launch command/environment and passes a runtime/execution fingerprint to every scorer.
4. Starts checksum-verified native JEV with the deterministic overrides and offline Hugging Face mode. No model downloads or package installs occur in the run script.
5. Verifies native identity/calibration and runs **20 charts twice per option schema** before predictions. Every pending job also repeats 20 charts. Any maximum probability difference **>0.01 aborts**. Reports now include per-row drift and argmax agreement; argmax stability never overrides a failed probability gate.
6. Scores frozen jobs at concurrency/batch one, preserving scope metadata, probabilities, predictions and latency. Timeout remains 120 seconds with three retries. Resume rejects changed input, model, question, context, concurrency/batch or execution fingerprint and skips completed rows.
7. Writes `results/inference/frozen_v3/<job>.csv` plus metadata/parity/error/summary files; session logs go to `shared/results/inference_run_v3/`. Stops its own server, **not the existing cloud pod or billing**.

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

The prior **46–121-minute estimate is withdrawn** because it assumed concurrency four and optimized cached/compiled execution. The corrected serial plan uses an explicit planning range of **103.2–309.6 minutes (about 103–310)**: 0.6–1.8 seconds per decide call plus 5–15 minutes for startup/validation. Experiment 03's historical median was 594.704 ms, but larger images and eager/cacheless batch-invariant kernels may cost more. The failed session does not measure corrected throughput. Hardware/driver changes and beta kernel support can move time outside this range; update the estimate only after a separately authorized successful gate.

## Later evaluation

Evaluate **06a test rows** (`--data-origin market --split test`) and **synthetic 06b** (`--data-origin synthetic --split eval`) separately. Retain all correlated versions within template blocks. The evaluator rejects mixed origins, showcase-only rows and accuracy-invalid time mirrors. `--clear-only` is secondary 06a analysis; report full-set results alongside it.

Real 06b is for visual examples and detector-agreement reporting only, regardless of train/test annotation. Do not report its per-class recognition accuracy. Align style/framing parents through frozen provenance; different PNG filenames are not independent observations. Horizontal controls report sensitivity rather than accuracy. User visual review remains pending; later dataset revisions must be separately recorded.
