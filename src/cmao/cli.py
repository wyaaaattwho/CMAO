from __future__ import annotations

import argparse
import json

from .pipeline import (
    run_advantage,
    run_analyze_cases,
    run_prepare_train_data,
    run_report,
    run_rerank_eval,
    run_sample,
    run_score,
    run_train,
    save_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CMAO offline experimentation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="Sample grouped CoTs from a model.")
    sample.add_argument("--config", required=True)
    sample.add_argument("--output", required=True)

    score = subparsers.add_parser("score", help="Score grouped CoTs.")
    score.add_argument("--input", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--config")

    advantage = subparsers.add_parser("advantage", help="Compute CMAO advantages.")
    advantage.add_argument("--input", required=True)
    advantage.add_argument("--output", required=True)
    advantage.add_argument("--config")

    rerank_eval = subparsers.add_parser("rerank_eval", help="Evaluate reranking strategies.")
    rerank_eval.add_argument("--input", required=True)
    rerank_eval.add_argument("--output", required=True)

    report = subparsers.add_parser("report", help="Print a stored report or build one from groups.")
    report.add_argument("--input", required=True)
    report.add_argument("--output")

    analyze_cases = subparsers.add_parser("analyze_cases", help="Export representative case studies.")
    analyze_cases.add_argument("--input", required=True)
    analyze_cases.add_argument("--output-prefix", required=True)

    prepare_train_data = subparsers.add_parser(
        "prepare_train_data",
        help="Flatten advantaged groups into policy training JSONL records.",
    )
    prepare_train_data.add_argument("--input", required=True)
    prepare_train_data.add_argument("--output", required=True)

    train = subparsers.add_parser("train_policy", help="Run CMAO LoRA policy training.")
    train.add_argument("--config", required=True)
    train.add_argument("--input", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "sample":
        run_sample(args.config, args.output)
        return
    if args.command == "score":
        run_score(args.input, args.output, args.config)
        return
    if args.command == "advantage":
        run_advantage(args.input, args.output, args.config)
        return
    if args.command == "rerank_eval":
        run_rerank_eval(args.input, args.output)
        return
    if args.command == "report":
        if args.output:
            save_report(args.input, args.output)
            print(f"Saved report to {args.output}")
        else:
            print(json.dumps(run_report(args.input), indent=2, ensure_ascii=False))
        return
    if args.command == "analyze_cases":
        result = run_analyze_cases(args.input, args.output_prefix)
        print(f"Saved case records to {result['case_path']}")
        print(f"Saved case summary to {result['summary_path']}")
        return
    if args.command == "prepare_train_data":
        summary = run_prepare_train_data(args.input, args.output)
        print(f"Saved training records to {args.output}")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    if args.command == "train_policy":
        summary = run_train(args.config, args.input)
        print(f"Saved training summary to {summary['output_dir']}/training_summary.json")
        return

    parser.error(f"Unknown command: {args.command}")
