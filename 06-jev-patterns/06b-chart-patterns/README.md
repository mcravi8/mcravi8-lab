# Experiment 06b: synthetic chart-pattern recognition

Day 1 prepares an evaluation-only image set for **autotrust/JEV-27B-VL**, the Hugging Face model used by the earlier experiments. No model inference, GPU session, or Runpod resource is started here. Real-market pattern detection is Day 2 work.

The [frozen generator protocol](protocol.json) was written before generating images or inspecting model outputs. There are 200 seeded clean templates, each rendered clean and with five increasing noise levels, for 1,200 images. Every image is 1024 x 640, with 60 to 120 daily-style OHLCV bars, candlesticks, volume, and no ticker, price, date, class, or other text. The independent sampling unit is `template_id`: the six versions of one template must remain together in future confidence intervals and comparisons.

| Source class | Clean templates | Noisy variants | Total |
|---|---:|---:|---:|
| Head and Shoulders | 29 | 145 | 174 |
| Inverse Head and Shoulders | 29 | 145 | 174 |
| Double Top | 29 | 145 | 174 |
| Double Bottom | 29 | 145 | 174 |
| Triangle | 28 | 140 | 168 |
| Bull Flag | 28 | 140 | 168 |
| None | 28 | 140 | 168 |
| **All** | **200** | **1,000** | **1,200** |

There are no rare or missing synthetic classes. These constructed counts do not estimate market prevalence. Triangle merges symmetric, ascending, and descending constructions. None consists of monotonic straight or gently curved paths, so it is a deliberately simple sanity control rather than a representative sample of market nonpatterns.

The clean close path interpolates explicit geometric pivots. Head and Shoulders and Double Top have similar outer peaks, then a completed neckline/trough break; their inverse classes invert prices. Triangles have converging upper/lower bounds and an exit. Bull Flag has an upward pole, a smaller downward parallel channel, then an upward break. Ten percent of bars precede and follow the formation, with breakout retained throughout the right margin. Price scale, amplitude, bar count, and volume vary with a fixed seed. The generator provides exact **source labels**; it does not claim detector agreement or that every heavily perturbed image stays unambiguous.

Noise amplitudes are `0.015, 0.035, 0.07, 0.12, 0.20` times the clean close range. Each template reuses one Brownian-bridge noise realization at all five amplitudes. Noise is zero at defining pivots and in the final post-breakout margin. This creates a paired severity ladder while retaining the construction's principal pivots and completed breakout. Severe noise can produce incidental local shapes. Report results by noise level and use template-group bootstrap blocks, not the 1,200 correlated images as independent examples.

[Review grid](results/samples/synthetic_grid.png) shows one template per class at noise 0, 0.035, and 0.20. Captions occur only in the review grid. The seven separate clean examples are `results/samples/sample_01.png` through `sample_07.png`; scored charts contain no captions.

![Clean and noisy synthetic patterns](results/samples/synthetic_grid.png)

From the repository root, after creating the separate Experiment 06 environment described in the parent README:

```sh
06-jev-patterns/.venv/bin/python 06-jev-patterns/06b-chart-patterns/src/synth.py
06-jev-patterns/.venv/bin/python -m unittest discover -s 06-jev-patterns/tests -p test_synthetic.py -v
```

This exports ignored `data/bars/*.csv`, ignored `charts/*.png`, and the scorer's ignored `charts/index.csv`. The committed `results/synthetic_index.csv` duplicates that index for review; `results/label_counts.json` and `results/render_manifest.json` record counts, hashes, source pivots, breaks, seeds, and paired template IDs. Dates in bar CSVs are artificial coordinates required by the shared OHLCV interface; `ticker=SYNTH`, the scorer index's `end_date` is blank, and no market date inference is valid.

The prepared fixed-100-bar framing control reuses each template and noise realization, interpolated onto exactly 100 bars. It holds bar count constant without inventing a second market window. Generate it in a separate ignored directory:

```sh
06-jev-patterns/.venv/bin/python 06-jev-patterns/06b-chart-patterns/src/synth.py \
  --fixed-bars 100 --output-dir 06-jev-patterns/06b-chart-patterns/data/fixed_100
```

This control is prepared, not generated in the Day 1 primary set. The real-data Day 2 control must instead use genuine fixed historical windows. Optional line/OHLC style variants and horizontal mirrors can use the shared renderer with the saved bars.

**Mirror interpretation correction:** the preserved planning document suggests that horizontal mirroring exchanges Head and Shoulders with Inverse and Double Top with Double Bottom. A horizontal flip reverses time; it does not invert prices and does not imply either class swap. A vertical price inversion creates those counterparts. Treat horizontal mirroring as a temporal-order intervention, with no assumed remapped ground truth.

Day 2 needs the real-data detector's supported patterns and license verified, an independent zigzag implementation, agreement and exclusive-label filtering, completed-pattern and overlap checks, date split with the three-month embargo, per-class hand review, and 80 to 120 real examples per class if available. Freeze any revised detector rules before scoring. Only after an explicit GPU authorization should inference parity and the zero-shot/context arms run, with all template variants grouped for inference. Synthetic results alone will establish visual recognition behavior, not predictive or trading skill.
