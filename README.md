# mcravi8-lab: AI x finance experiments

Experiments testing whether **AutoTrust's [JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL)** can forecast stock-price direction from charts, using model assets from **Hugging Face** and GPU compute on **Runpod**, alongside simpler CPU baselines. Experiment 06 also tests chart-pattern recognition. Code, forecasts, evaluation methods and negative results are documented together.

[![CPU research checks](https://github.com/mcravi8/mcravi8-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/mcravi8/mcravi8-lab/actions/workflows/tests.yml)

## Model and infrastructure

| Component | Role in these experiments |
| --- | --- |
| **JEV-27B-VL** (`autotrust/JEV-27B-VL`) | Vision-language decision model used for chart forecasts in Experiments 01-03, the LoRA fine-tune in Experiment 04 and pattern recognition in Experiment 06. |
| **[Hugging Face](https://huggingface.co/autotrust/JEV-27B-VL)** | Source of the model weights, System 1 decision adapter, calibration files, serving code and model card. The LoRA workflow uses Transformers and PEFT. |
| **[Runpod](https://www.runpod.io/)** | GPU hosting for historical JEV inference and LoRA training. Recorded weekly inference and training runs used NVIDIA H100 80GB GPUs; the weekly inference runs used `US-MO-1`. |
| **vLLM + JEV System 1** | The self-hosted server exposes `POST /v1/decide` on port 8000. Chart images and options `["Up", "Flat", "Down"]` produce one probability per option. |

See the [Experiment 01 serving instructions](01-jev-tsla-charts/README.md#future-pod-serving-setup) for the Runpod setup and the [Experiment 04 README](04-jev-lora-and-cnn/README.md#reproducible-training-preparation) for the LoRA workflow. Experiment 06 has recorded real JEV main-arm inference on Runpod and evaluated it on CPU; its secondary controls stopped at a repeatability gate. Experiment 05's numerical baselines and the validation suite run on CPU; the Experiment 05 results below are from that CPU study.

## Latest findings

**Experiment 06 shows useful but uneven pattern recognition.** With definitions, JEV scores **50.3% accuracy / 0.458 macro-F1** on 600 held-out candle charts and **55.8% / 0.492** on 1,200 fixed-100 synthetic charts from 200 independent parents. Definitions improve accuracy over zero-shot by 4.33 and 11.00 percentage points, respectively. Evening Stars, candle None, Triangles and Bull Flags remain weak; clean synthetic Bull Flags score 0/28. The v7 run collected 7,800 main predictions, then failed its first paraphrase repeatability gate, leaving prompt/style/placebo robustness untested. Real market chart patterns remain showcase-only. See [Experiment 06 results, baselines, intervals and limitations](06-jev-patterns/README.md#v7-results-useful-recognition-substantial-class-gaps).

**The tested models have not established a forecasting advantage.** The latest CPU study improved the research controls, but its numerical models underperformed a simple forecast based on training-period class frequencies.

Experiment 05 uses 12 tickers with adjusted price history back to 2000, or the IPO where later. It evaluates 17 annual development folds from 2010 through 2026; 2026 is partial. Each model fits the preceding five years, with training targets purged before the next test input period. The table compares the pooled training-frequency prior with logistic regression using chart geometry plus known return and volatility context.

| Target | Test observations | Prior accuracy | Logistic accuracy | Prior Brier | Logistic Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Up / Flat / Down | 7,502 | 40.46% | 38.00% | 0.660360 | 0.671732 |
| Up / Down | 7,490 | 55.46% | 53.24% | 0.495978 | 0.508527 |

Lower Brier is better. The context model improved Brier in only 2 of 17 years for each target. Both tasks cover 718 retained forecast dates; observations from the same date are correlated. These are exploratory development results, not independent confirmation or evidence that all charts and models are uninformative. See [Experiment 05](05-cpu-walkforward/README.md) for all baselines, confidence intervals, annual results and limitations.

The historical LoRA placebo conclusion was also corrected: 0 wins, 5 losses and 487 ties across five discordant dates give an exact paired p-value of **0.0625**. The earlier claim that LoRA was significantly below its mirrored-image placebo was unsupported. See [Experiment 04](04-jev-lora-and-cnn/README.md).

## Experiments

| Experiment | Question | Current status |
| --- | --- | --- |
| [01: TSLA and NVDA](01-jev-tsla-charts/README.md) | Can JEV predict from 20-session charts? | Historical real predictions, placebo and ticker probe collected. |
| [02: Weekly charts](02-jev-weekly-charts/README.md) | Does one-week chart geometry help? | Historical analysis corrected for paired and serial dependence. |
| [03: Twelve tickers](03-jev-12-tickers/README.md) | Does pooling or calibration help? | Reused historical test period; no demonstrated added skill. |
| [04: LoRA and CNN](04-jev-lora-and-cnn/README.md) | Does task-specific training help? | Historical results retained; revised training requires parity and checkpoint checks. |
| [05: CPU walk-forward study](05-cpu-walkforward/README.md) | Can numerical models beat training-frequency priors over longer histories? | CPU study complete; future protocol and model frozen. |
| [06: Pattern recognition](06-jev-patterns/README.md) | Can JEV identify candlestick and chart patterns? | V7 main predictions, CPU baselines and evaluation recorded; definitions help overall, major class gaps remain. Secondary controls blocked by a parity failure; 150 real parents are showcase-only. |

## Completed review and implementation

- Corrected evaluation uses training-selected references, paired accuracy and Brier comparisons, and moving-block intervals with 1/2/4/8 retained-week sensitivity. Test-majority benchmarks are descriptive only.
- Scoring supports separate binary and explicit volatility-based Flat tasks, blank and mirrored image controls, and model-identity checks that prevent incompatible runs from being mixed on resume. Historical predictions remain unchanged.
- Revised training validates data bundles and splits, uses representative fit-only parity fixtures, and blocks LoRA training above a 0.01 absolute HF-vLLM probability discrepancy. Checkpoints record provenance and validate reload inputs.
- The longer CPU study includes simple priors, momentum/reversal rules, numerical feature comparisons, a shuffled-label control, and a fixed future protocol.
- Experiment 06 adds 50-name data validation, candle labels with a visual-clarity audit and stricter analysis subset, varied synthetic pattern/noise controls, generalized native scoring with a mandatory repeatability gate, and multiclass evaluation. All **129 Experiment 06 CPU tests passed locally**, including fixed numeric shortcut checks on two separate cohorts, independent real-pattern detectors, split/overlap gates, showcase evaluation guards, subset/call-budget checks and mocked GPU-runbook failure handling.
- The **60 historical CPU tests passed** again in the Experiment 06 environment, including checkpoint round trips and localhost fake-server scoring. [GitHub CPU CI](https://github.com/mcravi8/mcravi8-lab/actions/workflows/tests.yml) runs all five suites on each push.

## Pending work and future holdout

Experiment 06's remaining paraphrase, style and placebo controls are blocked by the user-run v7 paraphrase gate: mean drift 0.002267, max 0.028662, raw argmax agreement 19/20, with one mismatch outside the allowed near-tie margins. The approved **v7 rule remains mean absolute option drift<=0.01 and every mismatch must have top-two margin<=0.01 in either pass**. It was adapted post-hoc after v2–v6. The [results noise-floor report](06-jev-patterns/shared/results/README.md) includes the v7 global gate's flagged max drift of 0.052885 and allowed-mismatch counts. Runtime/model/data pins remain frozen: serial batch one, eager/cacheless execution, no batch invariance for GDN_ATTN, PyPI Torch 2.13.0/torchvision 0.28.0, CUDA 13.0 and driver >=580. HF-vLLM parity, revised training and fresh-process LoRA equivalence remain unverified. Recognition results do not establish forecasting skill. No GPU or Runpod pod was started for this results review.

The 2025-2026 test period informed successive experiments, so results from 02-04 are exploratory. Experiment 05 reserves future chart weeks ending **2026-10-16 through 2027-10-08**. Its [protocol](05-cpu-walkforward/protocol.json) was pushed in [commit `0d2fae5`](https://github.com/mcravi8/mcravi8-lab/commit/0d2fae5) before inspecting the new CPU results. The primary model and comparator are frozen; future forecast collection and evaluation have not started. Success requires a positive paired Brier improvement with the prespecified four-week 95% interval wholly above zero.

## Run the checks

From the repository root, use Python 3.11 and install the research dependencies. On Linux, install the CPU-only PyTorch wheel for the checkpoint test:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install --only-binary=TA-Lib -r 06-jev-patterns/requirements.txt
.venv/bin/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m unittest discover -s 01-jev-tsla-charts/tests -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s 04-jev-lora-and-cnn/tests -v
.venv/bin/python -m unittest discover -s 05-cpu-walkforward/tests -v
.venv/bin/python -m unittest discover -s 06-jev-patterns/tests -v
```

On macOS, install `torch==2.14.1` from the default package index instead; the checkpoint test explicitly uses CPU. Install PyTorch for the complete suite, including Experiment 06's CPU CNN checks. Run each suite separately because historical experiments share module names. The suite comprises 31 pipeline/scoring/protocol tests, 8 statistical tests, 13 training tests and 8 baseline tests. Tests use synthetic OHLCV, localhost HTTP fixtures and CPU tensors, without downloading model weights. Experiment 06 adds pattern, rendering, arbitrary-option scoring and multiclass evaluation tests. GitHub Actions runs all five suites on pushes and pull requests.

## Reproducibility

Experiment 05's data cutoff is **2026-10-09**; Experiment 06 preserves its initial universe cutoff of **2026-09-30**. Experiment READMEs contain the corresponding data, rendering, scoring and evaluation commands. Regenerated market data, chart inputs, model weights, checkpoints and secrets are excluded from Git; results, checksums and selected figures are committed. Preserve local input snapshots and the frozen prospective model: a fresh clone does not contain them, and Yahoo's adjusted history can change between downloads.

The ticker universe is hand selected, the data are retrospective, and the studies do not test trading execution, transaction costs or P&L.
