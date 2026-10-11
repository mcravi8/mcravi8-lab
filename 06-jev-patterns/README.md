# Experiment 06: Can JEV-27B-VL recognize chart patterns?

**V7 main-arm inference and CPU evaluation are complete; secondary controls are blocked.** The user-run Runpod session collected 7,800 JEV predictions across ten zero-shot/definitions jobs, then stopped at the first paraphrase repeatability gate. The prepared datasets contain 1,200 candle charts, 1,200 synthetic charts and 150 real showcase-only parents. [PLAN.md](PLAN.md) preserves the original design; the results below describe the completed portion.

This experiment studies recognition of visible patterns. It uses **AutoTrust's [JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL)**, with model weights, calibration and native System 1 serving code from Hugging Face, served through vLLM on Runpod. V7 records Hugging Face revision `3514d1d8ead2189c75f433fa949fe63c8e3f3c66`. Local tests and the review of recorded results use CPU only. Recognition results do not establish future-return forecasting skill or trading value.

## V7 results: useful recognition, substantial class gaps

Definitions improve the aggregate scores in both tasks. JEV recognizes several reversal shapes well, but it misses most Evening Stars, candle negatives, Triangles and Bull Flags. These results support a narrower conclusion: **useful recognition of some classes, with incomplete robustness evidence**.

### Held-out market candles

Only the **600 test charts (75 per class)** enter the primary candle metrics. The 600 training charts were also scored, but are excluded from this table and used only to fit CPU baselines. Training ends in 2018; test input windows begin after the three-month 2019 embargo.

| Model / arm | Accuracy | Macro-F1 | Brier ↓ | ECE (15 bins) ↓ |
| --- | ---: | ---: | ---: | ---: |
| JEV zero-shot | 46.0% | 0.386 | 0.687 | 0.098 |
| JEV definitions | **50.3%** | **0.458** | **0.631** | 0.079 |
| Numeric logistic | 42.8% | 0.415 | 0.719 | 0.103 |
| CPU CNN | 34.3% | 0.328 | 0.803 | 0.028 |
| Uniform probabilities / training prior | 12.5% | 0.028 | 0.875 | 0.000 |

[Recorded candle metrics and figures](06a-candles/results/eval_v7). Definitions get **302/600** right versus **276/600** for zero-shot: **+4.33 percentage points**, paired ticker-quarter bootstrap 95% interval **+0.97 to +7.72 points**. The paired macro-F1 gain is **+0.072** (interval +0.035 to +0.108); Brier improves by 0.056. The definitions accuracy interval is **46.4–54.2%**. These exploratory intervals resample 352 ticker-quarter blocks, condition on the fixed predictions, and have no multiplicity correction. JEV pretraining overlap with these retrospective charts has not been assessed; the temporal split controls local baseline fitting.

Definitions beat the tested small logistic/CNN baselines on point estimates. Zero-shot has higher accuracy than logistic but **lower macro-F1**, illustrating why accuracy alone is insufficient. The CNN uses 477 fit rows, 120 time-ordered validation rows and three purged rows; this is a small fixed baseline, not an extensively tuned supervised comparison. Low ECE for CNN or uniform probabilities does not imply useful discrimination. Uniform argmax ties always select the first option, explaining its low macro-F1.

The [definitions confusion matrix](06a-candles/results/eval_v7/definitions_vs_zero_shot/confusion.png) shows the largest failures: **Evening Star 4/75 correct (5.3% recall)** and **None 6/75 (8.0%)**. Morning Stars are often called Bullish Engulfing (46/75); Evening Stars are often called Shooting Star (36/75) or Bearish Engulfing (30/75). These are failures against the frozen numerical labels. The detectors share a rulebook, and the AI-assisted audit found 10/50 subtle examples; independent expert review remains necessary to distinguish visual/definition ambiguity from model errors.

On the frozen **178-chart clear test subset**, accuracy is 52.8% with definitions versus 50.0% zero-shot. The paired accuracy interval is **−2.40 to +8.15 points**, so that subset does not establish an accuracy gain. It has a different class mix and must not replace the primary balanced test.

### Synthetic chart patterns

Each framing has **1,200 charts from the same 200 independent parents**, with clean plus five noise levels. Variable and fixed-100 images are paired views, not additional independent examples.

| Framing / arm | Accuracy | Macro-F1 | Brier ↓ | ECE (15 bins) ↓ |
| --- | ---: | ---: | ---: | ---: |
| Variable, zero-shot | 43.8% | 0.344 | 0.656 | 0.137 |
| Variable, definitions | 53.5% | 0.465 | 0.586 | 0.100 |
| Fixed 100, zero-shot | 44.8% | 0.352 | 0.650 | 0.140 |
| Fixed 100, definitions | **55.8%** | **0.492** | **0.567** | **0.093** |

[Recorded synthetic metrics and figures](06b-chart-patterns/results/eval_v7). Definitions improve accuracy by **9.75 points** in variable framing (paired template-bootstrap 95% interval **+6.58 to +13.25**) and **11.00 points** in fixed framing (**+7.67 to +14.58**). The fixed definitions accuracy interval is **49.8–61.5%**. All noise versions stay together in each bootstrap draw. Nominal row-level McNemar p-values are not valid independent evidence for these correlated versions; use the template intervals.

The aggregate score hides pronounced class imbalance in recognition:

| Source class | Fixed-100 definitions correct / total | Recall |
| --- | ---: | ---: |
| Head and Shoulders | 147/174 | 84.5% |
| Inverse Head and Shoulders | 95/174 | 54.6% |
| Double Top | 173/174 | 99.4% |
| Double Bottom | 168/174 | 96.6% |
| Triangle | 14/168 | 8.3% |
| Bull Flag | 5/168 | 3.0% |
| None | 68/168 | 40.5% |

Double Top/Bottom are also overpredicted: their recalls do not mean near-perfect precision. **No clean Bull Flag is correctly classified (0/28)**, so that weakness cannot be attributed solely to added noise. Review those errors and the generator's geometry independently before changing labels or prompts.

Accuracy by noise amplitude (200 charts per cell, same parents in every row):

| Noise / clean range | Variable zero-shot | Variable definitions | Fixed-100 zero-shot | Fixed-100 definitions |
| --- | ---: | ---: | ---: | ---: |
| 0 (clean) | 49.5% | 63.0% | 50.5% | 66.0% |
| 0.035 | 48.5% | 62.0% | 50.5% | 64.0% |
| 0.08 | 49.0% | 60.0% | 48.5% | 61.5% |
| 0.16 | 44.5% | 54.5% | 47.0% | 58.0% |
| 0.30 | 37.5% | 45.0% | 39.5% | 48.0% |
| 0.55 | 33.5% | 36.5% | 33.0% | 37.5% |

The noise breakdown is a descriptive review of the committed CSVs. At strong noise, labels describe the source construction even when the pattern is no longer visually clear. Chance under random uniform guessing is 1/7 (14.3%); the deterministic uniform argmax baseline predicts Head and Shoulders and scores 14.5%.

The synthetic logistic result (**18.25% accuracy**) is a **cross-domain comparator**: it fitted just 82 real showcase training examples, including 50 None, 22 Bull Flags and only one each of Double Top, Double Bottom and Triangle. It did not train on independent synthetic parents. Its weakness cannot establish superiority over a well-trained synthetic recognizer. The separate None-shortcut audit is a generator check, not a seven-class benchmark baseline.

### Confidence, completion and interpretation

At the prespecified confidence threshold **0.7**, candle definitions retain **160/600 (26.7%)** with **75.6% accuracy**; fixed-100 synthetic definitions retain **426/1,200 (35.5%)** with **90.4% accuracy**. These are selective results on different class mixtures, not overall accuracy or a calibrated deployment guarantee. At 0.9 the synthetic arm has 122/122 correct, but those related chart variants are not 122 independent trials. No new calibration was fitted.

The [v7 session](shared/results/inference_run_v7/session_summary.json) records **8,320 successful decide requests**: 7,800 main predictions plus 520 repeatability calls, with no HTTP failures, over **60.5 minutes elapsed**. The full 9,820-call schedule was not completed. The first candle paraphrase gate failed with mean drift **0.002267**, max **0.028662** and raw argmax agreement **19/20**; the mismatch had margins 0.011542/0.011556, outside the frozen 0.01 exception. No paraphrase predictions or style, horizontal-mirror or price-inversion controls were collected. This failure is repeatability within one prompt, not a measured disagreement between prompt wordings.

The gate was adapted post-hoc after v2–v6. V7's global candle gate passed under that approved rule despite a flagged **0.052885 max drift** on the recurring chart #15; two main-arm gates used one allowed near-tie mismatch each. See the [full noise-floor report](shared/results/README.md). Passing does not establish deterministic inference or HF-vLLM equivalence.

All **150 real pattern parents remain showcase-only**, including both framings and both main arms. Their predictions are available for qualitative review; no aggregate/per-class accuracy claim is permitted. The reported real-data result remains the [detector-agreement table](06b-chart-patterns/results/detector_agreement.csv).

The CPU forecasting bridge below does not meet its continuation criteria. Further recognition controls would address the separate recognition-benchmark question; they remain blocked by the frozen live gate. Any prompt, label, threshold or generator changes informed by these results need new held-out evaluation data. No GPU or Runpod resource was started to review the results or update this README.

## CPU forecasting bridge: no incremental five-day benefit

The [frozen protocol](config/forecast_bridge_v1.json) and [CPU implementation](shared/src/forecast_bridge.py) were pushed in [commit f780579](https://github.com/mcravi8/mcravi8-lab/commit/f780579) **before inspecting this return evaluation**. This separate study forecasts next-five-session Up/Flat/Down outcomes, using the one-third trailing five-session volatility band from Experiment 01. C uses saved JEV **pattern probabilities as inputs to a trained logistic forecasting model**; these results are not direct JEV return predictions or pattern-recognition scores.

Of the original 1,200 market candle rows, **1,182 are eligible: 584 train and 598 test**. Fourteen lack the 65 closes needed for the trailing scale, one lacks a complete forward target, and three training targets cross the 2018 boundary. The same eligible rows enter every arm. Both detectors reproduce the saved pattern labels; synthetic and real showcase rows are excluded.

| Forecast model | Held-out accuracy | Held-out Brier ↓ | Walk-forward accuracy | Walk-forward Brier ↓ |
| --- | ---: | ---: | ---: | ---: |
| Training-frequency prior | **39.6%** | **0.658871** | **39.6%** | **0.659010** |
| A: numeric OHLCV | 34.8% | 0.776384 | 32.6% | 0.760502 |
| B: A + detector indicators | 32.3% | 0.793376 | 33.4% | 0.764061 |
| C: A + JEV probabilities | 33.4% | 0.790343 | 33.3% | 0.770202 |

The primary model fits only pre-2019 training observations; secondary expanding walk-forward models refit annually for 2019–2026. Training targets are purged before the earliest test information history, including the 65-close volatility lookback. Scalers fit on training rows only; all 27 checkpoint reload checks pass.

C's paired Brier improvement versus A is **−0.013959**, with an exploratory four-retained-week bootstrap 95% interval **[−0.024287, −0.003695]**: adding JEV worsens the primary score. Annual walk-forward gives **−0.009699 [−0.018551, −0.000451]**, and C improves Brier in only **1/8 years**. B also worsens held-out Brier by 0.016992; its walk-forward interval includes zero. All fitted models underperform the training-frequency prior.

**All three frozen continuation conditions fail.** This study supports pausing further JEV compute for the tested five-day forecasting route. It does not establish that all patterns, chart representations or models are uninformative: the cohort is small, balanced by recognition class and retrospective, and the comparator is a fixed high-dimensional linear model with no hyperparameter search. Existing inference noise and unknown JEV pretraining overlap also remain limitations. No executable trading costs or P&L were evaluated.

[Full results and reproduction](shared/results/forecast_bridge_v1/README.md), [metrics and uncertainty](shared/results/forecast_bridge_v1/metrics.json), [paired-comparison figure](shared/results/forecast_bridge_v1/comparison.png), and [independent validation](shared/results/forecast_bridge_v1/validation.json) preserve the outcome. The audit independently checks all 1,182 targets and ten model/evaluation summaries, plus nine purged folds and 27 checkpoint checks. All **140 Experiment 06 tests pass locally**. This run uses CPU only, makes no HTTP/GPU/Runpod calls or data downloads, and leaves Experiment 05's prospective protocol and frozen models unchanged.

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

Before any pending predictions, **20 deterministically selected charts are scored twice**. Require **mean absolute option drift<=0.01** and **every argmax mismatch to have top-two margin<=0.01 in either pass**; all other charts must match. Mean includes all 20 × K comparisons without exclusions. Reports retain raw agreement, per-pass/either-pass near-tie counts, allowed/disallowed mismatches, max/p90 and chart-max flags>0.03. Ties select the first declared option. Matteo approved this final v7 exception **after the v6 failure**; see the [post-hoc policy history](GPU_RUNBOOK.md#failed-gate-diagnosis-and-decision). Failure blocks scoring and is logged. Pending resumes repeat the live gate; `--parity-only` always performs 40 calls, while a completed no-op resume makes none.

This gate checks repeatability. It does **not** prove Hugging Face/vLLM implementation parity or model accuracy; the separate Experiment 04 HF-vLLM gate is still required before its training workflow.

Scoring writes index metadata plus `probabilities` (JSON array in option order), `prediction`, and `latency_ms`. Adjacent `.meta.json`, `.parity.json`, `.summary.json`, and `_errors.csv` files record provenance, gate results, progress and failures. Resume skips valid predictions, repairs an interrupted final CSV record, refuses mismatched input/model/context provenance, and locks an output against concurrent writers. Errors return a nonzero exit status and can be retried by rerunning the same command. Each arm needs its own output path.

Supported CLI/environment configuration: `--base-url` / `JEV_BASE_URL`, `--served-model-name` / `JEV_SERVED_MODEL`, `--model-revision` / `JEV_MODEL_REVISION`, `--batch-size` / `JEV_BATCH_SIZE` (16), `--concurrency` / `JEV_CONCURRENCY` (4), `--timeout` / `JEV_TIMEOUT` (60 seconds), `--retries` / `JEV_RETRIES` (3). Optional authentication uses `JEV_API_KEY`. Remote endpoints must use HTTPS. Controls are `--control image|mirror|blank`.

For local contract validation, start `tests/fake_jev_server.py` in another terminal. `--dry-run` only permits loopback and marks its outputs synthetic; fake results are not model results. The test suite exercises 20-chart parity, ten predictions, partial resume, complete no-op resume, drift, retries, provenance mismatch, aliases, more than 16 options and damaged CSV recovery.

## Evaluation and CPU baselines

`shared/src/evaluate.py` reports accuracy, macro-F1 over every declared class, a confusion matrix, multiclass Brier (sum over classes), log loss, 15-bin top-confidence ECE, reliability plots, and error risk/coverage at confidence 0.5/0.7/0.9. Empty accepted sets have null risk, not zero. All argmax ties use the first ordered option.

Real-data intervals resample whole **ticker-calendar-quarter** blocks. Synthetic intervals group all noise/framing versions by `template_id`. Missing or insufficient blocks yield explicit unavailable intervals. Paired comparisons align exact file IDs and truth, use the same bootstrap draw for both arms, and report exact two-sided McNemar. Its independence assumption is stronger than the block bootstrap's: correlated discordant rows invalidate its nominal p-value. Ticker-quarter intervals also omit cross-ticker common shocks and dependence lasting longer than a quarter. These are exploratory intervals, with no multiplicity correction.

`shared/src/baselines.py` supplies training-majority, training-frequency prior, uniform probabilities, numerical logistic regression and the Experiment 04 CNN architecture with an arbitrary output count. Logistic uses only normalized visible OHLCV interpolated to 32 positions, with its scaler fitted on training rows. CNN validation comes from the latest 20% of training dates with overlapping fit windows purged. It verifies checkpoint reload predictions. Synthetic examples cannot train fitted baselines; uniform needs no training data. Default CLI mode is a preflight; fitting needs explicit `--mode train`.

Recognition-baseline training dates end by **2018-12-31**. January–March 2019 is embargoed, and test input windows begin on or after **2019-04-01**. Synthetic data are evaluation-only. The train set is for baseline fitting and any future calibration; held-out labels must not select prompts, thresholds, hyperparameters or examples.

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

All **140 Experiment 06 tests passed locally** in the pinned environment, including eleven forecasting-bridge tests. The 60 historical tests remain part of GitHub CI. `pip check` passed. The Day 1 images and both sample collections were validated and visually inspected. The trimmed freeze checks all 4,400 unique target PNGs, their matching numeric bars and all 50 raw input hashes; see [day2_validation.json](shared/results/day2_validation.json). All seven real review grids and selected derived styles were inspected during Day 2; existing reference boards are inactive historical artifacts. The data cache verifier confirmed all 50 inputs without network access and preserved the frozen input-manifest hash. [GitHub CPU CI](https://github.com/mcravi8/mcravi8-lab/actions/workflows/tests.yml) runs on each staged push.

The candle definitions contain 157 words; chart-pattern definitions contain 255 words. They describe the frozen detector/generator rules. No real-model prediction, real-dataset baseline fit, GPU or Runpod call was made for Day 1.


- [06a candles](06a-candles/README.md): TA-Lib plus independent pandas labels, exact agreement, exclusive labels, ticker/year-stratified selection, twenty-candle images and a fixed visual audit.
- [06b synthetic](06b-chart-patterns/README.md): 200 clean templates and five paired noise levels (1,000 noisy images), seven classes, sixty to 120 bars, 1024 x 640 images.

Day 2 real detection is complete across all 50 input snapshots: **150 showcase-only parents (82 train / 68 test)**, with 6 H&S, 2 inverse, 1 Double Top, 1 Double Bottom, 4 Triangle, 36 Bull Flag and 100 None. All positive classes are rare; inverse and both doubles have no test examples. See the [real protocol, license, agreement rates and review grids](06b-chart-patterns/README.md#day-2-real-market-dataset). Fixed-100 historical frames are paired with variable frames, and user visual review remains pending. The [frozen arms](config/inference_arms.json) and [future GPU runbook](GPU_RUNBOOK.md) are now prepared. All 4,400 active target PNGs are hash-pinned with indexes and code. All 150 real parents remain in both main arms and framings, marked `showcase_only` / `accuracy_valid=false`; no aggregate or per-class model accuracy claims are permitted. The evaluator rejects their accuracy metrics. The [reported detector-agreement table](06b-chart-patterns/results/detector_agreement.csv) is a CPU detector-overlap result, not model accuracy or expert label validity.

One geometric correction to the preserved plan: a **horizontal mirror reverses time**; it does not exchange Head and Shoulders with Inverse or Double Top with Double Bottom. Price inversion creates those counterparts. Horizontal controls therefore have no assumed class swap. The current shared mirror is horizontal.

The trimmed frozen schedule has **21 jobs, 9,820 decide calls**: 8,900 predictions plus 920 parity calls. Corrected serial execution has a planning estimate of **103–310 GPU minutes** (103.2–309.6 before rounding). These are planning assumptions, not measured throughput; retries/resumes add calls. See the [budget and future command](GPU_RUNBOOK.md).

The **reference-image arm is removed**. Main zero-shot/definitions arms use one primary wording; the second wording runs only on **100 charts per sub-part**, for both arms. Style and horizontal-mirror subsets also have 100 charts each: test-only market candles for 06a and distinct clean synthetic parents for 06b. OHLC/line and mirror controls are zero-shot only, and candle style reuses primary predictions. An additional 100-parent eligible synthetic subset tests price-inversion class flips. Secondary subsets are fixed before inference and do not measure noisy-chart robustness. Variable/fixed-100 main collections remain paired: genuine historical bars for real showcases, same-parent interpolation for synthetic charts.

V7 main zero-shot/definitions scoring, CPU baselines and held-out arm comparisons are recorded above. Calibration fitting and all secondary robustness controls remain pending because the first paraphrase gate blocked the rest of the user-run session. Real 06b predictions remain descriptive showcases. The frozen configuration records the plan as prepared before execution; [session summaries](shared/results/inference_run_v7/session_summary.json) and per-job outputs record actual progress. Earlier failed attempts remain in [diagnostics](shared/results/diagnostics/2026-10-10/diagnosis.json). Tests and detector agreement alone establish neither recognition accuracy nor trading value.


## Parity correction after the first GPU attempt

The user-run candle gate produced 3/20 probability differences over 0.01 (maximum 0.033344), with 20/20 unchanged argmaxes and no HTTP errors. The gate was already serial: prediction concurrency four did not run. Prefix-cache cold/warm paths and BF16/LoRA kernel reductions are plausible causes; a cu129 vLLM wheel paired with Torch CUDA 13.0 remains a build risk, without a matched-build comparison.

The gate history is post-hoc: after v4, Matteo approved max drift<=0.03 plus full argmax agreement; after v5 max 0.0412173, he approved mean<=0.01 plus full agreement. V6 failed only on argmax agreement **19/20**, with **mean 0.00260983 and max 0.02224329**. On 2026-10-11 Matteo approved the final **v7 near-tie exception**: a mismatch is allowed only if either pass has top-two margin<=0.01. Mean<=0.01 remains mandatory. V6 has one near-tie chart in pass 2 and one allowed mismatch; raw agreement remains 19/20. See [all v2–v6 numbers and counts](shared/results/README.md). Actual v3 failed startup and has no drift measurement.

Serial concurrency/batch one, eager execution, no prefix caching/chunked prefill/async scheduling, seed zero and all runtime/model/data settings remain frozen. Batch invariance stays removed for GDN_ATTN. The nightly **0.31.1rc1.dev253+g7d0b4e57a.cu129**, PyPI Torch **2.13.0**/CUDA **13.0**, torchvision **0.28.0**, driver **>=580**, and explicit Linux scorer **/root/svenv/bin/python** requirements are unchanged.

The full call plan remains **9,820** with **frozen_v7** outputs; the recorded partial run used 8,320 calls in 60.5 minutes elapsed. The 103–310-minute figure above is the original planning assumption. Both global gates and ten main-arm gates passed live under v7; the first paraphrase gate failed and the remaining jobs were not run. [Original failures](shared/results/diagnostics/2026-10-11-v6-parity/diagnosis.json) and earlier approvals remain preserved. Pending scoring still requires fresh live gates. No GPU was started for this results review.
