"""CPU-only provenance, parity and accumulation helpers for Experiment 04."""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np

LABELS = ["Up", "Flat", "Down"]
PROBS = ["p_up", "p_flat", "p_down"]
PARITY_ATOL = 0.01  # maximum absolute probability error; fixed before any new run
RELOAD_ATOL = 1e-7
GPU_PINS = {"torch": "2.14.1", "torchvision": "0.29.1", "transformers": "5.19.0",
            "peft": "0.21.2", "safetensors": "0.8.0", "accelerate": "1.15.0"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_path(root: Path, relative: str) -> Path:
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Path escapes artifact directory: {relative}")
    return path


def hash_files(root: Path, paths) -> dict[str, str]:
    return {str(path): sha256(safe_path(root, str(path))) for path in sorted(set(paths))}


def verify_hashes(root: Path, hashes: dict[str, str]) -> None:
    if not hashes:
        raise ValueError("Missing artifact hashes")
    for relative, expected in hashes.items():
        if sha256(safe_path(root, relative)) != expected:
            raise ValueError(f"Artifact hash mismatch: {relative}")


def source_hashes(root: Path) -> dict[str, str]:
    root = Path(root)
    paths = [str(path.relative_to(root)) for path in (root / "src").glob("*.py")]
    paths += [str(path.relative_to(root)) for path in root.glob("requirements*.txt")]
    return hash_files(root, paths)


def runtime() -> dict:
    packages = {}
    for name in sorted({*GPU_PINS, "numpy", "pandas", "pillow", "pyarrow"}):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    installed = {distribution.metadata["Name"]: distribution.version
                 for distribution in metadata.distributions() if distribution.metadata["Name"]}
    result = {"python": sys.version, "platform": platform.platform(), "packages": packages,
              "installed_packages": dict(sorted(installed.items()))}
    result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def require_gpu_packages(environment: dict) -> None:
    errors = [f"{name}: expected {version}, found {environment['packages'].get(name)}"
              for name, version in GPU_PINS.items()
              if (environment["packages"].get(name) or "").split("+")[0] != version]
    if sys.version_info[:2] != (3, 11):
        errors.append("Python 3.11 is required")
    if errors:
        raise ValueError("Unsupported candidate training environment: " + "; ".join(errors))


def model_hashes(root: Path) -> dict[str, str]:
    root = Path(root)
    required = ["config.json", "calibration.json", "adapter_vllm/decision_head.json",
                "adapter_vllm/adapter_config.json", "adapter_vllm/adapter_model.safetensors"]
    for path in required:
        if not safe_path(root, path).is_file():
            raise ValueError(f"Missing local model artifact: {path}")
    paths = [str(path.relative_to(root)) for path in root.rglob("*")
             if path.is_file() and path.suffix in {".json", ".safetensors", ".model", ".txt", ".py"}
             and not any(part.startswith(".") for part in path.relative_to(root).parts)]
    if not any(p.endswith(".safetensors") and not p.startswith("adapter_vllm/") for p in paths):
        raise ValueError("Missing local base-model weights")
    return hash_files(root, paths)


def probability_array(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError("Expected nonempty N x 3 probabilities")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    if not np.allclose(values.sum(axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("Probabilities must sum to one")
    return values / values.sum(axis=1, keepdims=True)


def parity_report(actual, reference, tolerance: float = PARITY_ATOL) -> dict:
    if not np.isfinite(tolerance) or not 0 < tolerance <= PARITY_ATOL:
        raise ValueError(f"Parity tolerance must be positive and at most {PARITY_ATOL}")
    actual, reference = probability_array(actual), probability_array(reference)
    if actual.shape != reference.shape:
        raise ValueError("Parity fixture shapes differ")
    error = np.abs(actual - reference)
    return {"n": len(actual), "atol": tolerance, "max_abs_diff": float(error.max()),
            "mean_abs_diff": float(error.mean()), "passed": bool(error.max() <= tolerance)}


def require_parity(actual, reference, tolerance: float = PARITY_ATOL) -> dict:
    report = parity_report(actual, reference, tolerance)
    if not report["passed"]:
        raise ValueError(f"HF-vLLM parity failed: {report['max_abs_diff']:.6g} > {tolerance}; training blocked")
    return report


def accumulation_groups(order, size: int):
    """Each optimizer step averages its actual group, including the final remainder."""
    if size <= 0:
        raise ValueError("Accumulation size must be positive")
    for start in range(0, len(order), size):
        yield order[start:start + size]
