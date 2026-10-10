# AI x finance experiments

Small, reproducible AI x finance experiments with honest results.

| Experiment | Question | Current status |
| --- | --- | --- |
| [01: TSLA and NVDA](01-jev-tsla-charts/README.md) | Can JEV predict from 20-session charts? | Historical real predictions, placebo and ticker probe collected. |
| [02: Weekly charts](02-jev-weekly-charts/README.md) | Does one-week chart geometry help? | Historical analysis corrected for paired and serial dependence. |
| [03: Twelve tickers](03-jev-12-tickers/README.md) | Does pooling or calibration help? | Exploratory reused test period; no demonstrated added skill. |
| [04: LoRA and CNN](04-jev-lora-and-cnn/README.md) | Does task-specific training help? | Historical results retained; future training now requires parity and checkpoint checks. |
| [05: CPU walk-forward study](05-cpu-walkforward/README.md) | Can simple numerical models beat train-only priors over longer histories? | 17 annual development folds completed; future protocol and model frozen. |

The tested models have not established a forecasting advantage. Experiment 05's numerical models also underperformed the pooled training-frequency baseline. This is evidence about these specific inputs, targets and recipes, not proof that all charts or models are uninformative.

The 2025-2026 test period informed successive experiments, so results from 02-04 are exploratory. Experiment 05 reserves future chart weeks from 2026-10-16 through 2027-10-08; those outcomes have not been evaluated. Its protocol was pushed in commit `0d2fae5` before inspecting the new CPU results.

## Checks

From the repository root, in a Python 3.11 environment with `05-cpu-walkforward/requirements.txt` installed:

```sh
python -m unittest discover -s 01-jev-tsla-charts/tests -v
python -m unittest discover -s tests -v
python -m unittest discover -s 04-jev-lora-and-cnn/tests -v
python -m unittest discover -s 05-cpu-walkforward/tests -v
```

Run each suite separately because historical experiments share module names. The checkpoint test additionally needs PyTorch 2.14.1; GitHub Actions installs its CPU-only wheel. Tests use synthetic OHLCV, localhost HTTP fixtures and CPU tensors. They do not download model weights or start a GPU. The workflow runs on pushes and pull requests.

Regenerated market data, chart inputs, model weights, checkpoints and secrets are excluded from Git. Results, checksums and selected figures are committed. Keep the local input snapshots: Yahoo's adjusted history can change, so a fresh download is not guaranteed to reproduce their hashes.
