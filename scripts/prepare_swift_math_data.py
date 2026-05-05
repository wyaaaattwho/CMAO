from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmao.datasets import load_problems

DEFAULT_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def _build_rows(args: argparse.Namespace, split: str) -> list[dict[str, object]]:
    problems = load_problems(
        dataset_name=args.dataset_name,
        split=split,
        limit=args.limit,
        path=args.path,
        config_name=args.config_name,
    )
    rows: list[dict[str, object]] = []
    for index, problem in enumerate(problems):
        prompt = problem.prompt.strip()
        if args.instruction and args.instruction not in prompt:
            prompt = f"{prompt} {args.instruction}"
        rows.append(
            {
                "messages": [{"role": "user", "content": prompt}],
                "solution": problem.gold_answer,
                "gold_answer": problem.gold_answer,
                "problem_id": problem.id,
                "source": problem.source,
                "split": split,
                "index": index,
            }
        )
    return rows


def _write_jsonl(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare math data in ms-swift messages JSONL format.")
    parser.add_argument("--dataset-name", default="math-500")
    parser.add_argument("--config-name")
    parser.add_argument("--train-split", default="test")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--path", help="Optional local JSON/JSONL problem file.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--train-output", default="data/swift_math500/train.jsonl")
    parser.add_argument("--val-output", default="data/swift_math500/val.jsonl")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    args = parser.parse_args()

    train_rows = _build_rows(args, args.train_split)
    val_rows = _build_rows(args, args.val_split)
    _write_jsonl(train_rows, ROOT / args.train_output)
    _write_jsonl(val_rows, ROOT / args.val_output)
    print(f"Wrote {len(train_rows)} train rows to {args.train_output}")
    print(f"Wrote {len(val_rows)} val rows to {args.val_output}")


if __name__ == "__main__":
    main()
