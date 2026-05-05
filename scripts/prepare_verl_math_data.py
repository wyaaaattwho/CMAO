from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmao.datasets import load_problems


DEFAULT_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def _build_rows(args: argparse.Namespace, split: str) -> list[dict]:
    problems = load_problems(
        dataset_name=args.dataset_name,
        split=split,
        limit=args.limit,
        path=args.path,
        config_name=args.config_name,
    )
    rows = []
    for index, problem in enumerate(problems):
        prompt = problem.prompt.strip()
        if args.instruction and args.instruction not in prompt:
            prompt = f"{prompt} {args.instruction}"
        rows.append(
            {
                "data_source": problem.source,
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": problem.gold_answer},
                "extra_info": {
                    "split": split,
                    "index": index,
                    "problem_id": problem.id,
                    "source": problem.source,
                },
            }
        )
    return rows


def _write_parquet(rows: list[dict], output: Path) -> None:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("Install `datasets` to write verl parquet files.") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CMAO math data in verl RLHF parquet format.")
    parser.add_argument("--dataset-name", default="math-500")
    parser.add_argument("--config-name")
    parser.add_argument("--train-split", default="test")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--path", help="Optional local JSON/JSONL problem file.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--train-output", default="data/verl_math500/train.parquet")
    parser.add_argument("--val-output", default="data/verl_math500/test.parquet")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    args = parser.parse_args()

    train_rows = _build_rows(args, args.train_split)
    val_rows = _build_rows(args, args.val_split)
    _write_parquet(train_rows, ROOT / args.train_output)
    _write_parquet(val_rows, ROOT / args.val_output)
    print(f"Wrote {len(train_rows)} train rows to {args.train_output}")
    print(f"Wrote {len(val_rows)} val rows to {args.val_output}")


if __name__ == "__main__":
    main()
