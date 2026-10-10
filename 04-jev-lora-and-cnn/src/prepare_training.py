"""Export verified chart identities, splits, references and fit-only parity fixtures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from jev_api import DEFAULT_HORIZON, DEFAULT_VOL_LOOKBACK, DEFAULT_WINDOW_SESSIONS, task_label, task_question
from splits import ROOT, assign_fit_split, representative_fixtures
from training_utils import PROBS, hash_files, probability_array, runtime, safe_path, sha256, source_hashes, verify_hashes, write_json


TASK_SETTINGS = {"window_sessions": DEFAULT_WINDOW_SESSIONS, "horizon": DEFAULT_HORIZON,
                 "vol_lookback": DEFAULT_VOL_LOOKBACK, "flat_band": 1 / 3}


def prepare(root: Path, output: Path, reference: Path, task="historical", reference_metadata=None) -> dict:
    root, output, reference = Path(root).resolve(), Path(output).resolve(), Path(reference).resolve()
    if output == root or output.is_relative_to(root / "charts"):
        raise ValueError("Training export must not replace the experiment or source charts")
    index_path = root / "charts/index.csv"
    idx = assign_fit_split(pd.read_csv(index_path))
    if task not in ("historical", "ternary-volatility"):
        raise ValueError("This three-class trainer supports historical or ternary-volatility tasks")
    metadata_hash = None
    if task != "historical":
        if reference_metadata is None:
            raise ValueError("Corrected task requires fresh matching reference metadata")
        reference_metadata = Path(reference_metadata)
        info = json.loads(reference_metadata.read_text())
        if info.get("task") != task or info.get("task_settings") != TASK_SETTINGS:
            raise ValueError("Reference task/prompt settings differ from training")
        if info.get("synthetic") or info.get("control") != "image":
            raise ValueError("Corrected-task reference must be real, unmodified image predictions")
        if not (info.get("server_identity") or {}).get("model_revision"):
            raise ValueError("Corrected-task reference requires a pinned server model revision")
        for relative in idx.file:
            if (info.get("image_sha256") or {}).get(relative) != sha256(safe_path(root, relative)):
                raise ValueError(f"Reference chart hash missing or mismatched: {relative}")
        for row in idx.to_dict("records"):
            if task_label(row, task, TASK_SETTINGS["flat_band"]) != row["label"]:
                raise ValueError(f"Index label differs from the corrected task: {row['file']}")
            question_hash = hashlib.sha256(task_question(row, task, **TASK_SETTINGS).encode()).hexdigest()
            if (info.get("question_sha256") or {}).get(row["file"]) != question_hash:
                raise ValueError(f"Reference question hash missing or mismatched: {row['file']}")
        metadata_hash = sha256(reference_metadata)
    refs = pd.read_csv(reference)
    fields = ["file", "ticker", "end_date", "label"]
    if not set(fields + PROBS).issubset(refs.columns) or refs.file.duplicated().any():
        raise ValueError("Reference predictions require unique file/ticker/date/label and three probabilities")
    joined = idx[fields].merge(refs[fields + PROBS], on="file", suffixes=("", "_ref"),
                             how="left", validate="one_to_one")
    for column in fields[1:]:
        if not joined[column].equals(joined[column + "_ref"]):
            raise ValueError(f"Reference prediction identity mismatch: {column}")
    joined[PROBS] = probability_array(joined[PROBS])
    fixture_rows = representative_fixtures(idx)
    fixture_fields = ["file", "ticker", "end_date", "label", "fit_split"]
    if "sigma" in fixture_rows:
        fixture_fields.append("sigma")
    fixtures = fixture_rows[fixture_fields].merge(
        joined[["file"] + PROBS], on="file", validate="one_to_one")
    output.mkdir(parents=True, exist_ok=True)
    for relative in idx.file:
        source, target = safe_path(root, relative), safe_path(output, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    idx.to_csv(output / "index.csv", index=False, float_format="%.17g")
    joined[["file"] + PROBS].to_csv(output / "zs_preds.csv", index=False, float_format="%.17g")
    fixtures.to_csv(output / "parity_fixtures.csv", index=False, float_format="%.17g")
    manifest = {"schema_version": 1, "index_source_sha256": sha256(index_path),
                "reference_source_sha256": sha256(reference), "reference_source": str(reference),
                "task": task, "task_settings": TASK_SETTINGS,
                "reference_metadata_sha256": metadata_hash,
                "historical_reference_model_identity_unverified": task == "historical",
                "fixture_selection": "fit only; every ticker, available class, first/middle/last date",
                "counts": {str(k): int(v) for k, v in idx.fit_split.value_counts().items()},
                "n_parity_fixtures": len(fixtures), "runtime": runtime(),
                "source_sha256": source_hashes(root),
                "files_sha256": hash_files(output, ["index.csv", "zs_preds.csv", "parity_fixtures.csv", *idx.file])}
    write_json(output / "manifest.json", manifest)
    return manifest


def verify_bundle(root: Path):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("task") not in ("historical", "ternary-volatility") or manifest.get("task_settings") != TASK_SETTINGS:
        raise ValueError("Unsupported bundle task/prompt settings")
    verify_hashes(root, manifest["files_sha256"])
    idx = pd.read_csv(root / "index.csv")
    canonical = assign_fit_split(idx)
    if not idx.fit_split.equals(canonical.fit_split):
        raise ValueError("Exported fit_split differs from the canonical split")
    fixtures = pd.read_csv(root / "parity_fixtures.csv")
    expected = representative_fixtures(canonical)
    if set(fixtures.file) != set(expected.file) or fixtures.file.duplicated().any():
        raise ValueError("Parity fixtures are not the canonical representative fit-only set")
    reference = pd.read_csv(root / "zs_preds.csv")
    if reference.file.duplicated().any() or set(reference.file) != set(idx.file):
        raise ValueError("Reference coverage differs from chart index")
    merged = fixtures.merge(reference, on="file", suffixes=("", "_ref"), validate="one_to_one")
    for column in PROBS:
        if not (merged[column] == merged[column + "_ref"]).all():
            raise ValueError("Parity fixture/reference probabilities differ")
    probability_array(reference[PROBS])
    return idx, fixtures, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "data/training_bundle")
    parser.add_argument("--reference", type=Path, default=ROOT / "results/predictions.csv")
    parser.add_argument("--task", choices=("historical", "ternary-volatility"), default="historical")
    parser.add_argument("--reference-metadata", type=Path)
    args = parser.parse_args()
    manifest = prepare(args.root, args.out, args.reference, args.task, args.reference_metadata)
    print(json.dumps({"counts": manifest["counts"], "n_parity_fixtures": manifest["n_parity_fixtures"]}, indent=2))


if __name__ == "__main__":
    main()
