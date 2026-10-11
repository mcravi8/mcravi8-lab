# Experiment 06 trimmed inference runbook

**Approved v7 near-tie parity policy prepared on CPU; not GPU verified.** A prior on-pod session failed the first global candle gate after 40 decide calls and wrote no predictions. A subsequent v3 startup failed because JEV GDN_ATTN rejects batch invariance; no decide calls ran in that session. The v4 serial/eager/cacheless gate then failed at maximum drift 0.0184107, despite 20/20 matching top predictions. The user-run v5 gate subsequently failed at maximum drift 0.0412173 on chart #15, with 20/20 matching top predictions. V6 then failed only on raw argmax agreement (19/20), despite mean drift 0.0026098 and max 0.0222433. Matteo explicitly approved the final v7 near-tie exception after observing this failure. [run_gpu.py](scripts/run_gpu.py) is the single future entry point. It does not create, start, stop or contact Runpod resources. Inference requires a separately authorized, already prepared GPU host.

## Frozen assets and arms

The [model contract](config/model_contract.json) pins **autotrust/JEV-27B-VL** to Hugging Face snapshot **3514d1d8ead2189c75f433fa949fe63c8e3f3c66**, including all 19 base/adapter safetensor files, serving/calibration SHA-256s and tokenizer/config Git blob IDs. No model weights were downloaded or GPU started during this correction.

The [pinned model card](https://huggingface.co/autotrust/JEV-27B-VL/blob/3514d1d8ead2189c75f433fa949fe63c8e3f3c66/README.md) and [native server](https://huggingface.co/autotrust/JEV-27B-VL/blob/3514d1d8ead2189c75f433fa949fe63c8e3f3c66/serve_decide.py) define native System 1 `/v1/decide`, `thinking="off"`, `strategy="single"`, with ordered eight- or seven-class options. Every scheduled call has one target image. **The reference-image arm is removed**, including its enable flag; existing CPU reference boards are inactive historical artifacts.

[Inference config](config/inference_arms.json), schema 7, freezes **zero-shot and definitions** with one primary question wording per task. A second wording is confined to **100 charts per sub-part**, for both arms. These rows reuse main-arm predictions for the primary wording. No prompt or subset is selected from prediction performance.

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

**Historical decision:** initially retain the predeclared 0.01 tolerance while changing execution. V5 changed to max drift <=0.03 plus matching argmaxes after v4; v6 replaced its maximum criterion with mean <=0.01 after v5; the active v7 rule below adds the approved near-tie exception after v6.

The subsequent [v3 startup evidence](shared/results/diagnostics/2026-10-10-gdn-attn/diagnosis.json), copied from `shared/results/inference_run_v3/server.log`, establishes **`VLLM batch_invariant mode is not supported for GDN_ATTN.`** The flag exists in vLLM, but JEV’s attention backend cannot use it. The engine exited before readiness or scoring. **Remove batch invariance from the frozen policy** and explicitly clear `VLLM_BATCH_INVARIANT` inherited from the shell; this applies to the server and CLI-help probe. The remaining serial/eager/cacheless settings are a repeatability attempt, not a deterministic guarantee.

The [preserved v4 paired responses and diagnostics](shared/results/diagnostics/2026-10-10-v4-parity/diagnosis.json) confirm **maximum probability drift 0.018410737935632693**, **5/20 rows above 0.01**, and **20/20 identical top predictions**. All 40 decide requests returned HTTP 200; no prediction jobs started. Runtime and server logs show BF16, serial batch-1 scoring, eager execution, disabled prefix caching/chunked prefill, and no batch-invariance flag.

The [preserved v6 paired responses and diagnostics](shared/results/diagnostics/2026-10-11-v6-parity/diagnosis.json) confirm **mean drift 0.0026098296378638183**, max 0.022243287166081593, and **19/20 raw argmax matches**. All 40 requests returned HTTP 200; no predictions ran. Gate chart #11, `charts/7d763e0bb9783f355926.png`, changed from Doji to Shooting Star. Its first-pass top probabilities were 0.4546401571 and 0.4408969473 (margin 0.0137432098); its second-pass top probabilities were 0.4485482060 and 0.4485217768 (margin 0.0000264292). The second pass meets the approved near-tie definition.

| Recorded attempt | Mean option drift | Max option drift | P90 option drift | Raw argmax matches | Near-tie charts, either pass | Original outcome |
|---|---:|---:|---:|---:|---:|---|
| v2 original gate | 0.0019776218 | 0.0333443688 | 0.0056115729 | 20/20 | 0 | Failed max<=0.01 |
| v3 startup | N/A | N/A | N/A | N/A | N/A | GDN_ATTN rejected batch invariance before scoring |
| v4 serial/eager/no-cache | 0.0020260185 | 0.0184107379 | 0.0059701163 | 20/20 | 0 | Failed max<=0.01 |
| v5 serial/eager/no-cache | 0.0024581249 | 0.0412173047 | 0.0057232281 | 20/20 | 0 | Failed max<=0.03 |
| v6 serial/eager/no-cache | 0.0026098296 | 0.0222432872 | 0.0076997699 | 19/20 | 1 (pass 2 only) | Mean<=0.01 passed; strict argmax agreement failed |

**Active v7 policy, approved by Matteo on 2026-10-11:** score the same 20 charts twice; require **mean absolute probability drift<=0.01** and **every argmax mismatch to have a top-two margin<=0.01 in at least one pass**. A chart with margins>0.01 in both passes must match argmax. This exception applies to any qualifying chart, without a filename whitelist or chart exclusion.

Mean still includes **all 20 × K chart-option entries** (K=8 candles, K=7 patterns):

`mean_abs_diff = sum(abs(pass 1[i,k] - pass 2[i,k])) / (20 * K)`

For each pass, `top_two_margin = largest_probability - second_largest_probability` over all options, after unchanged probability validation/normalization. The comparison is inclusive (`<=0.01`), and **either** pass suffices; exact ties qualify. Prediction tie-breaking still selects the first declared option. No probabilities or predicted labels are rewritten to manufacture agreement. Mean drift remains mandatory even when every mismatch qualifies for the exception.

Reports retain **raw argmax agreement counts/rates and mismatch files**; they do not claim 20/20 agreement when an exception applies. They separately record per-pass/either-pass near-tie chart counts, each chart's two margins, allowed/disallowed mismatch counts/files, and the argmax condition outcome. Near-tied charts whose predictions match still enter the near-tie count, but not the allowed-mismatch count. Console gate results and job summaries also report these counts.

Mean/max/p90 drift, chart-level p90 and flags for per-chart max drift **strictly>0.03** remain unchanged. P90 option drift uses linear interpolation over 20 × K absolute differences; chart-level p90 uses 20 per-chart maxima. Flags do not exclude charts or fail the gate on their own.

**This is the third post-hoc gate change, explicitly approved after observing failures.** The original rule was max<=0.01; after v4, v5 required max<=0.03 and full argmax agreement; after v5, v6 required mean<=0.01 and full agreement; after v6, v7 adds the conditional near-tie exception. The reason is that residual **BF16 kernel nondeterminism under serial, eager, no-cache settings** can reorder two almost equal option probabilities. This remains a working explanation, not an isolated proof of a particular kernel/build fault. V6’s second-pass margin was only 0.0000264 while overall mean drift stayed 0.00261. The approved policy distinguishes these ranking changes from mismatches with both margins beyond 0.01.

The [results noise-floor report](shared/results/README.md) records all v2–v6 measurements, near-tie counts and original failures, with [machine-readable evidence](shared/results/parity_noise_floor.json). The policy was adapted using this fixed candle fixture; these observations are not independent validation or a universal noise bound. Disclose the adaptations, residual probability drift and allowed mismatch counts alongside later probability/calibration results. Repeatability does not establish HF-vLLM equivalence or recognition accuracy.

**Prior chart #15 identity proof remains intact:** `charts/b5674ffc8efb3bc45e6d.png`, SHA-256 `0aed05d7b75aa6e4b93166f53e9e7a7544bb9c7ae6a00b9785c6b62caa492b70`. The original0.033344 run and v5 share its filename/position/hash; current bytes match. The earlier run is named **v2** on disk. Actual v3 failed startup and has no measured drift.

Original v2/v4/v5/v6 reports remain failures under their original policies. Saved pairs satisfy v7 in **CPU replay only**, which is not a new GPU gate. V7 outputs are frozen and **not run**; fresh live gates remain required. The snapshot of the prior v6 noise-floor report and settings is preserved with the v6 archive.

All other settings remain frozen; v7 uses:

- Scorer concurrency **1**, client batch size **1**, for both gates and predictions.
- **Batch invariance removed** because the pinned build rejects GDN_ATTN. Inherited flags are cleared before CLI help/server launch.
- `--enforce-eager`, `--no-enable-prefix-caching`, `--no-enable-chunked-prefill`, `--no-async-scheduling` and `--seed 0`. Token budget 16,384; GDN prefill `triton`.
- Native LoRA `jev-decision`, processed logprobs, raw template/calibrated System 1 payload and `--max-num-seqs 8` retained; read-out temperature untouched.
- `PYTHONHASHSEED=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `NVIDIA_TF32_OVERRIDE=0`.
- Fresh `frozen_v7` outputs and `inference_run_v7` session files; older cached gates cannot be resumed as v7.

**V7 has not run on a GPU.** Runtime packages, model snapshot, payload/calibration, environment, driver minimum, prompts, charts, subsets, batch/concurrency and server flags remain unchanged from v6. Config retains the approval history and records this as the final requested gate change. Any future change must be separately recorded.

## Observed build pins and Linux scorer

[GPU runtime contract](config/gpu_runtime.json) and [critical server requirements](requirements-jev-server.txt) pin the actually observed numerical stack:

| Component | Observed / required version |
|---|---|
| vLLM nightly | `0.31.1rc1.dev253+g7d0b4e57a.cu129` |
| Torch distribution | `2.13.0`, installed from PyPI |
| torchvision distribution | `0.28.0`, installed from PyPI |
| Torch compiled CUDA | **`13.0`**, confirmed by v3 `torch.version.cuda`; imported Torch is `2.13.0+cu130` |
| Triton | `3.7.1` |
| FlashInfer | `0.7.0.post1` |
| Transformers | `5.19.0` |
| NVIDIA CUDA runtime package | `13.0.96` |
| Runpod host NVIDIA driver | **580 or newer**; checked before importing Torch or starting the server |

The complete [original package inventory](shared/results/diagnostics/2026-10-10/runtime_lock.json) preserves the mixed CUDA 12/13 dependencies; the [v3 inventory](shared/results/diagnostics/2026-10-10-gdn-attn/runtime_lock.json) confirms Torch distribution `2.13.0`, imported `2.13.0+cu130`, CUDA `13.0` and torchvision `0.28.0`. Their PyPI installation source is reported by the user; an installed-version inventory cannot prove the wheel index. CUDA 12.9 is **not** asserted as Torch's build. Original wheel hashes/container digest remain unavailable, so these pins do not claim byte-identical reconstruction or binary compatibility. Keep the cached exact vLLM wheel; do not upgrade a moving nightly index.

**Select an already authorized Runpod host with NVIDIA driver 580 or newer.** Driver 570 pods fail with this CUDA 13.0 Torch build, as reported by the user. [NVIDIA’s compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) requires driver **>=580** for CUDA 13.x. Inspect the **Driver Version**, not the `CUDA Version` header: the latter is driver capability, not Torch's compiled build. This script uses `nvidia-smi --query-gpu=driver_version --format=csv,noheader` and rejects any version below 580, missing GPUs/query tool, malformed results or a failed query **before Torch import and server startup**. It records all driver versions in the runtime lock and rejects changes on resume. It does not upgrade a driver or provision a replacement pod.

For future preparation of the existing server environment, install the requested Torch versions explicitly from **PyPI**, using [requirements-jev-torch.txt](requirements-jev-torch.txt). Do not use the CUDA 12.9 PyTorch index for these packages. The latest supplied server interpreter is `/root/venv/bin/python3`; use that path only if its environment still matches the pins:

```sh
cd /root/repo
export JEV_SERVER_PYTHON=/root/venv/bin/python3
nvidia-smi --query-gpu=driver_version --format=csv,noheader
"$JEV_SERVER_PYTHON" -m pip install -r 06-jev-patterns/requirements-jev-torch.txt
"$JEV_SERVER_PYTHON" -m pip check
```

These are documented future-host commands; no server packages were installed during this CPU change. Installation and wheel sourcing remain separate from the GPU run script, which checks exact package versions and `torch.version.cuda=13.0` before launch.

A macOS virtual environment cannot be copied to Linux. Create or reuse the **on-pod Linux scorer ** in a separate environment, matching the actual `/root/svenv` inventory through [requirements-scoring.txt](requirements-scoring.txt). The lightweight scorer needs no Torch/vLLM. From the prepared pod's repository root:

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

Set `JEV_SERVER_PYTHON` to the actual Linux interpreter whose environment matches the observed server pins. The v3 diagnostics used `/root/venv/bin/python3`. Both interpreters must be explicit; there is no macOS venv fallback.

```sh
cd /root/repo
export JEV_SERVER_PYTHON=/root/venv/bin/python3
/root/svenv/bin/python 06-jev-patterns/scripts/run_gpu.py \
  --execute \
  --model-dir /workspace/models/JEV-27B-VL \
  --server-python "$JEV_SERVER_PYTHON" \
  --scorer-python /root/svenv/bin/python
```

There is no reference-image enable flag. Without `--execute`, the script validates local assets and prints the plan. **Neither mode was invoked during this CPU correction; tests import helpers and mock process launch.**

The script:

1. Verifies frozen assets, the under-10,000 call cap, showcase-only scope and model weight/serving/config hashes; rejects an occupied port.
2. Checks **NVIDIA driver >=580 before importing Torch **, then exact required server/scorer package pins, `torch.version.cuda=13.0` and supported CLI flags **before starting a server**. Runtime probes inspect metadata/import packages/CLI help only; they do not create an inference engine. No batch-invariance capability is required; the rejected environment flag is removed.
3. Records driver versions, packages, Torch installation source, build/platform metadata and policy in `shared/results/inference_run_v7/runtime_lock.json`; rejects changed runtimes on resume. It records the launch command/environment and passes a runtime/execution fingerprint to every scorer.
4. Starts checksum-verified native JEV with the serial/eager/cacheless overrides and offline Hugging Face mode. No model downloads or package installs occur in the run script.
5. Verifies native identity/calibration and runs **20 charts twice per option schema** before predictions. Every pending job also repeats 20 charts. Abort if **mean absolute probability drift>0.01 OR any argmax mismatch has top-two margins>0.01 in both passes**. Reports retain both passes, the v7 policy, mean/max/p90, flags over 0.03, raw argmax agreement, per-pass near-tie margins/counts and allowed/disallowed mismatches. Cached gates and resume metadata must match the active policy.
6. Scores frozen jobs at concurrency/batch one, preserving scope metadata, probabilities, predictions and latency. Timeout remains 120 seconds with three retries. Resume rejects changed input, model, question, context, concurrency/batch or execution fingerprint and skips completed rows.
7. Writes `results/inference/frozen_v7/<job>.csv` plus metadata/parity/error/summary files; session logs go to `shared/results/inference_run_v7/`. Stops its own server, **not the existing cloud pod or billing**.

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

The prior **46–121-minute estimate is withdrawn** because it assumed concurrency four and optimized cached/compiled execution. The corrected serial plan uses an explicit planning range of **103.2–309.6 minutes (about 103–310)**: 0.6–1.8 seconds per decide call plus 5–15 minutes for startup/validation. Experiment 03's historical median was 594.704 ms, but larger images and eager/cacheless kernels may cost more. The failed session does not measure corrected throughput. Hardware/driver changes and hybrid kernel support can move time outside this range; update the estimate only after a separately authorized successful gate.

## Later evaluation

Evaluate **06a test rows** (`--data-origin market --split test`) and **synthetic 06b** (`--data-origin synthetic --split eval`) separately. Retain all correlated versions within template blocks. The evaluator rejects mixed origins, showcase-only rows and accuracy-invalid time mirrors. `--clear-only` is secondary 06a analysis; report full-set results alongside it.

Real 06b is for visual examples and detector-agreement reporting only, regardless of train/test annotation. Do not report its per-class recognition accuracy. Align style/framing parents through frozen provenance; different PNG filenames are not independent observations. Horizontal controls report sensitivity rather than accuracy. User visual review remains pending; later dataset revisions must be separately recorded.
