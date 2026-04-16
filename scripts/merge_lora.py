from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a standalone model directory.")
    parser.add_argument("--adapter", required=True, help="Path to the saved LoRA adapter checkpoint.")
    parser.add_argument("--output", required=True, help="Directory for the merged standalone model.")
    parser.add_argument("--device-map", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Merging LoRA adapters requires peft and transformers.") from exc

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter,
        torch_dtype="auto",
        device_map=args.device_map,
    )
    merged = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model saved to {output_dir}")


if __name__ == "__main__":
    main()
