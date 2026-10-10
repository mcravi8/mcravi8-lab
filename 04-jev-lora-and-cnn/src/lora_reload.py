"""Reload a verified LoRA checkpoint for inference without downloading weights."""
import argparse
from pathlib import Path

from jev_training import load_trained
from lora_train import write_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", choices=("fit", "val", "test"), default="test")
    args = parser.parse_args()
    runner = load_trained(args.model_dir, args.data, args.checkpoint)
    # Use the exact index validated and bound to the checkpoint before model loading.
    rows = runner.verified_index[runner.verified_index.fit_split == args.split]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(args.out, rows, runner.score(rows))


if __name__ == "__main__":
    main()
