"""Local-only System 1 reconstruction shared by training and checkpoint reload.

Imports of GPU packages and allocation happen only inside load_system1().
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from jev_api import DIRECTION_QUESTION, task_question
from training_utils import model_hashes, require_gpu_packages, runtime, verify_hashes

PROMPT = ("[kind] choice\n[state] Chart: <|vision_start|><|image_pad|><|vision_end|>"
          f"\n[question] {DIRECTION_QUESTION}\n[options]\nA) Up\nB) Flat\nC) Down\n[decision]:")
TARGET_MODULES = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)$"


def chart_prompt(row, task="historical", task_settings=None):
    question = task_question(row, task, **(task_settings or {}))
    return ("[kind] choice\n[state] Chart: <|vision_start|><|image_pad|><|vision_end|>"
            f"\n[question] {question}\n[options]\nA) Up\nB) Flat\nC) Down\n[decision]:")


class System1:
    def __init__(self, model, processor, ids, bias, temperature, data_root, provenance):
        self.model, self.processor = model, processor
        self.ids, self.bias, self.temperature = ids, bias, temperature
        self.data_root, self.provenance = Path(data_root), provenance

    def logits(self, row, mirror=False):
        from training_utils import safe_path
        with Image.open(safe_path(self.data_root, row["file"])) as source:
            image = source.convert("RGB")
        if mirror:
            image = ImageOps.mirror(image)
        prompt = chart_prompt(row, self.provenance["task"], self.provenance["task_settings"])
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt",
                                add_special_tokens=False).to("cuda")
        output = self.model(**inputs, logits_to_keep=1)
        return (output.logits[0, -1, self.ids].float() + self.bias) / self.temperature

    def score(self, rows, mirror=False):
        import torch
        self.model.eval()
        with torch.no_grad():
            return np.array([torch.softmax(self.logits(row, mirror), 0).cpu().numpy()
                             for row in rows.to_dict("records")])


def load_system1(model_dir: Path, data_root: Path, expected_hashes=None,
                 task="historical", task_settings=None) -> System1:
    require_gpu_packages(runtime())
    if expected_hashes is not None:
        verify_hashes(model_dir, expected_hashes)
        hashes = expected_hashes
    else:
        hashes = model_hashes(model_dir)
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("System 1 requires a CUDA GPU with bfloat16 support; no training performed")
    model_dir = Path(model_dir)
    head = json.loads((model_dir / "adapter_vllm/decision_head.json").read_text())
    temperature = json.loads((model_dir / "calibration.json").read_text())["per_kind"]["choice"]
    start = head["slots"]["ranges"]["choice"][0]
    ids = head["verbalizer_ids"][start:start + 3]
    if len(ids) != 3 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid System 1 decision readout")
    bias = torch.tensor(head["bias"][start:start + 3], dtype=torch.float32, device="cuda")
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map="cuda", local_files_only=True)
    config = model.config
    tied = getattr(config, "tie_word_embeddings", False) or getattr(
        getattr(config, "text_config", config), "tie_word_embeddings", False)
    if tied:
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    adapter = load_file(model_dir / "adapter_vllm/adapter_model.safetensors")
    acfg = json.loads((model_dir / "adapter_vllm/adapter_config.json").read_text())
    if any(acfg.get(key) for key in ("use_rslora", "use_dora", "fan_in_fan_out", "modules_to_save")):
        raise ValueError("Unsupported original decision-adapter merge configuration")
    modules, merged = dict(model.named_modules()), []
    a_keys = [key for key in adapter if ".lora_A." in key]
    paired = {key for a in a_keys for key in (a, a.replace("lora_A", "lora_B"))}
    if not a_keys or set(adapter) != paired:
        raise ValueError("Unsupported original decision-adapter tensor layout")
    scale = acfg["lora_alpha"] / acfg["r"]
    with torch.no_grad():
        for key in a_keys:
            path = key.replace("base_model.model.", "", 1).split(".lora_A.")[0]
            candidates = [path, path.replace("model.layers.", "model.language_model.layers.", 1),
                          path.replace("model.", "model.language_model.", 1)]
            target = next((name for name in candidates if name in modules and hasattr(modules[name], "weight")), None)
            if target is None:
                raise ValueError(f"Cannot map decision-adapter tensor {key}")
            a = adapter[key].to("cuda", torch.float32)
            b = adapter[key.replace("lora_A", "lora_B")].to("cuda", torch.float32)
            weight = modules[target].weight
            delta = b @ a
            if delta.shape != weight.shape:
                raise ValueError(f"Decision-adapter shape mismatch at {target}")
            weight.copy_((weight.float() + scale * delta).to(weight.dtype))
            merged.append(target)
    provenance = {"model_sha256": hashes, "merge": "linear B@A, alpha/r, float32 sum then bfloat16",
                  "merged_modules": merged, "untied_lm_head": bool(tied), "historical_prompt": PROMPT,
                  "task": task, "task_settings": task_settings or {},
                  "choice_ids": ids, "temperature": temperature,
                  "runtime": runtime(), "cuda_version": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(), "dtype": "bfloat16"}
    return System1(model, processor, ids, bias, temperature, data_root, provenance)


def validate_reload(checkpoint: Path, data_root: Path):
    """Bind replay inputs, task and inference code before importing GPU packages."""
    from prepare_training import verify_bundle
    manifest = json.loads((Path(checkpoint) / "provenance.json").read_text())
    bundle_hashes = manifest.get("bundle_manifest_sha256")
    if not isinstance(bundle_hashes, dict) or set(bundle_hashes) != {"manifest.json"}:
        raise ValueError("Checkpoint lacks its recorded bundle manifest checksum")
    verify_hashes(data_root, bundle_hashes)
    index, _, bundle = verify_bundle(data_root)
    base = manifest.get("base", {})
    if base.get("task") != bundle["task"] or base.get("task_settings") != bundle["task_settings"]:
        raise ValueError("Checkpoint task/prompt settings differ from the verified bundle")
    verify_hashes(checkpoint, manifest["adapter_sha256"])
    inference_files = ["src/jev_training.py", "src/jev_api.py", "src/training_utils.py"]
    sources = manifest.get("source_sha256", {})
    if any(path not in sources for path in inference_files):
        raise ValueError("Checkpoint lacks its recorded inference source checksums")
    verify_hashes(Path(__file__).resolve().parents[1], {path: sources[path] for path in inference_files})
    return index, manifest


def load_trained(model_dir: Path, data_root: Path, checkpoint: Path) -> System1:
    """Reconstruct the original decision merge, then apply the verified new LoRA.

    The save-time check covers adapter serialization on an already merged base.
    Fresh-process GPU reconstruction remains unverified until a separate replay.
    """
    index, manifest = validate_reload(checkpoint, data_root)
    from peft import PeftModel
    runner = load_system1(model_dir, data_root, manifest["base"]["model_sha256"],
                          manifest["base"]["task"], manifest["base"]["task_settings"])
    runner.model = PeftModel.from_pretrained(runner.model, str(checkpoint), is_trainable=False)
    runner.verified_index = index
    return runner
