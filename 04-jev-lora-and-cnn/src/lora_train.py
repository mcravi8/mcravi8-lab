"""Fail-closed local-only LoRA pipeline. Default mode performs CPU preflight.

Parity uses representative fitting charts. Training requires parity first and
reports the same HF implementation before/after. No model is downloaded here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd

from jev_training import TARGET_MODULES, load_system1
from prepare_training import verify_bundle
from splits import ROOT
from training_utils import (LABELS, PARITY_ATOL, PROBS, RELOAD_ATOL, accumulation_groups,
                            hash_files, model_hashes, parity_report, require_gpu_packages,
                            runtime, source_hashes, verify_hashes, write_json)


def logloss(probabilities, labels):
    y = labels.map(LABELS.index).to_numpy()
    return float(-np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-9, 1)).mean()), float(
        (probabilities.argmax(1) == y).mean())


def write_predictions(path, rows, probabilities):
    pd.DataFrame({"file": rows.file, **dict(zip(PROBS, probabilities.T))}).to_csv(
        path, index=False, float_format="%.17g")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "parity", "train"), default="preflight")
    parser.add_argument("--data", type=Path, default=ROOT / "data/training_bundle")
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/JEV-27B-VL"))
    parser.add_argument("--out", type=Path, default=ROOT / "results/revised_training")
    parser.add_argument("--parity-atol", type=float, default=PARITY_ATOL)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--evals-per-epoch", type=int, default=2)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args(argv)
    if min(args.rank, args.lr, args.accum, args.epochs, args.evals_per_epoch, args.patience) <= 0:
        parser.error("Training hyperparameters must be positive")
    if not 0 < args.parity_atol <= PARITY_ATOL:
        parser.error(f"--parity-atol must be positive and at most {PARITY_ATOL}")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "train" and ((args.out / "lora_training.json").exists() or list(args.out.glob("lora_r*"))):
        raise ValueError("Training artifacts already exist; use a fresh --out directory")
    environment = runtime()
    write_json(args.out / "environment.json", environment)
    idx, fixtures, bundle = verify_bundle(args.data)
    require_gpu_packages(environment)
    # Hashing validates local files; preflight does not import torch or allocate a model.
    local_hashes = model_hashes(args.model_dir)
    provenance = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "source_sha256": source_hashes(ROOT), "bundle": bundle,
                  "runtime": environment, "candidate_gpu_environment_verified": False}
    write_json(args.out / "preflight.json", {**provenance, "model_sha256": local_hashes})
    if args.mode == "preflight":
        print("CPU preflight passed. CUDA execution and HF-vLLM parity remain unverified.")
        return
    runner = load_system1(args.model_dir, args.data, local_hashes, bundle["task"], bundle["task_settings"])
    actual = runner.score(fixtures)
    parity = parity_report(actual, fixtures[PROBS], args.parity_atol)
    write_predictions(args.out / "parity_hf_probs.csv", fixtures, actual)
    write_json(args.out / "parity.json", {**parity, "fixture_files": fixtures.file.tolist(),
                                        "base": runner.provenance, "source_sha256": provenance["source_sha256"]})
    if not parity["passed"]:
        raise ValueError(f"HF-vLLM parity failed: {parity['max_abs_diff']:.6g} > {args.parity_atol}; training blocked")
    if args.mode == "parity":
        print("Fit-only HF-vLLM parity passed. No training performed.")
        return
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    torch.manual_seed(0)
    random.seed(0)
    fit, val, test = (idx[idx.fit_split.eq(name)].reset_index(drop=True) for name in ("fit", "val", "test"))
    t0 = time.time()
    # These are diagnostic outputs, never used to select checkpoints or hyperparameters.
    baseline = {}
    for name, rows, mirror in (("val", val, False), ("test", test, False), ("test_placebo", test, True)):
        probabilities = runner.score(rows, mirror)
        write_predictions(args.out / f"hf_zero_shot_{name}_preds.csv", rows, probabilities)
        baseline[name] = pd.DataFrame({"file": rows.file, **dict(zip(PROBS, probabilities.T))})
    baseline["test"].merge(baseline["test_placebo"].rename(columns={p: p + "_pl" for p in PROBS}),
                            on="file", validate="one_to_one").to_csv(
        args.out / "hf_zero_shot_test_preds.csv", index=False, float_format="%.17g")
    runner.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    runner.model.enable_input_require_grads()
    runner.model = get_peft_model(runner.model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules=TARGET_MODULES))
    params = [p for p in runner.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0)
    steps_per_epoch = math.ceil(len(fit) / args.accum)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1., (step + 1) / 10) * max(0., 1 - step / total_steps))
    eval_steps = max(1, steps_per_epoch // args.evals_per_epoch)
    vl, va = logloss(runner.score(val), val.label)
    history = [{"examples_seen": 0, "optimizer_steps": 0, "epoch": 0., "train_loss": None,
                "val_logloss": vl, "val_acc": va}]
    best, best_at, best_state, bad = vl, history[0], None, 0
    seen, steps, run, nrun, stop = 0, 0, 0., 0, False
    lossf = torch.nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        order = list(range(len(fit)))
        random.Random(epoch).shuffle(order)
        runner.model.train()
        for group in accumulation_groups(order, args.accum):
            opt.zero_grad()
            for i in group:
                row = fit.iloc[i]
                loss = lossf(runner.logits(row)[None], torch.tensor([LABELS.index(row.label)], device="cuda"))
                (loss / len(group)).backward()
                run += loss.item()
                nrun += 1
                seen += 1
            torch.nn.utils.clip_grad_norm_(params, 1.)
            opt.step()
            sched.step()
            steps += 1
            if steps % eval_steps == 0 or seen == (epoch + 1) * len(fit):
                vl, va = logloss(runner.score(val), val.label)
                record = {"examples_seen": seen, "optimizer_steps": steps, "epoch": seen / len(fit),
                          "train_loss": run / nrun, "val_logloss": vl, "val_acc": va}
                history.append(record)
                print(json.dumps(record), flush=True)
                run, nrun = 0., 0
                if vl < best - 1e-4:
                    best, best_at, bad = vl, record, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in runner.model.state_dict().items() if "lora_" in k}
                else:
                    bad += 1
                    if bad >= args.patience:
                        stop = True
                        break
                runner.model.train()
        if stop:
            break
    if best_state is not None:
        runner.model.load_state_dict(best_state, strict=False)
    else:
        with torch.no_grad():
            for name, param in runner.model.named_parameters():
                if "lora_B" in name:
                    param.zero_()
    checkpoint = args.out / f"lora_r{args.rank}"
    before_reload = runner.score(fixtures)
    runner.model.save_pretrained(checkpoint)
    adapter_hashes = hash_files(checkpoint, [p.name for p in checkpoint.iterdir() if p.is_file()])
    checkpoint_provenance = {"base": runner.provenance, "source_sha256": provenance["source_sha256"],
                             "bundle_manifest_sha256": hash_files(args.data, ["manifest.json"]),
                             "adapter_sha256": adapter_hashes, "best": best_at,
                             "reload_verification": "adapter serialization on the same already merged base",
                             "fresh_process_reload_verified": False,
                             "reload_recipe": "load_system1(original local base + original decision merge), then PeftModel.from_pretrained(new adapter)"}
    write_json(checkpoint / "provenance.json", checkpoint_provenance)
    verify_hashes(checkpoint, adapter_hashes)
    # This verifies adapter serialization, not fresh-process reconstruction of the base.
    # unload() removes the NEW adapter without merging it, preserving the original decision merge.
    merged_base = runner.model.unload()
    runner.model = PeftModel.from_pretrained(merged_base, checkpoint, is_trainable=False)
    reload_error = float(np.abs(before_reload - runner.score(fixtures)).max())
    if reload_error > RELOAD_ATOL:
        raise ValueError(f"LoRA checkpoint reload probabilities differ by {reload_error}; outputs blocked")
    results = {}
    for name, rows, mirror in (("val", val, False), ("test", test, False), ("test_placebo", test, True)):
        probabilities = runner.score(rows, mirror)
        write_predictions(args.out / f"lora_{name}_preds.csv", rows, probabilities)
        results[name] = pd.DataFrame({"file": rows.file, **dict(zip(PROBS, probabilities.T))})
    combined = results["test"].merge(results["test_placebo"].rename(columns={p: p + "_pl" for p in PROBS}),
                                     on="file", validate="one_to_one")
    combined.to_csv(args.out / "lora_test_preds.csv", index=False, float_format="%.17g")
    output_paths = [p.name for p in args.out.glob("*.csv")]
    write_json(args.out / "lora_training.json", {**provenance, "base": runner.provenance, "parity": parity,
                                               "candidate_gpu_environment_verified": True,
                                               "best": best_at, "history": history, "seconds": time.time() - t0,
                                               "checkpoint": checkpoint.name, "reload_max_abs_diff": reload_error,
                                               "reload_verification": "adapter serialization on the same already merged base",
                                               "fresh_process_reload_verified": False,
                                               "output_sha256": hash_files(args.out, output_paths)})


if __name__ == "__main__":
    main()
