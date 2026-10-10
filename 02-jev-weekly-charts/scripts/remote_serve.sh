#!/usr/bin/env bash
# Run on an existing pod only, after installing the model-card-compatible vLLM.
# Copy this to ~/Desktop/runpod-setup/remote_serve.sh before transferring it.
# Upstream serve.sh preserves the System 1 LoRA and required max-num-seqs 8.
set -euo pipefail
export MODEL_DIR="${MODEL_DIR:-/workspace/models/JEV-27B-VL}"
export PORT="${PORT:-8000}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
for asset in serve.sh serve_decide.py adapter_vllm/decision_head.json calibration.json; do
  if [[ ! -f "$MODEL_DIR/$asset" ]]; then
    printf 'Missing cached model asset: %s/%s\n' "$MODEL_DIR" "$asset" >&2
    exit 1
  fi
done
exec bash "$MODEL_DIR/serve.sh"
