#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REPORT_SUFFIX = "_math500test_report.json"
ANALYSIS_SUFFIX = "_math500test_analysis_summary.json"
EXPECTED_MODELS_DEFAULT = (
    "math500_online_grpo_1.5b_lora",
    "math500_online_cmao_1.5b_lora",
    "math500_online_grpo_7b_lora",
    "math500_online_cmao_7b_lora",
    "qwen2.5-math-1.5b-base",
    "qwen2.5-math-7b-base",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{value * 100:.{digits}f}%"


def _f(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def _parse_model_tag(name: str) -> tuple[str, str]:
    if "-base" in name:
        method = "base"
    elif "_cmao_" in name:
        method = "cmao"
    elif "_grpo_" in name:
        method = "grpo"
    else:
        method = "unknown"
    size = "7b" if "_7b_" in name else ("1.5b" if "_1.5b_" in name else "unknown")
    return method, size


def _safe_get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _collect_rows(
    eval_dir: Path,
    train_dir: Path,
    report_suffix: str,
    analysis_suffix: str,
    expected_models: list[str] | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    row_map: dict[str, dict[str, Any]] = {}
    case_keys: set[str] = set()

    report_paths = sorted(eval_dir.glob(f"*{report_suffix}"))
    for report_path in report_paths:
        model_id = report_path.name[: -len(report_suffix)]
        report = _load_json(report_path)
        analysis_path = eval_dir / f"{model_id}{analysis_suffix}"
        analysis = _load_json(analysis_path) if analysis_path.exists() else {}
        train_summary_path = train_dir / model_id / "training_summary.json"
        train_summary = _load_json(train_summary_path) if train_summary_path.exists() else {}

        method, size = _parse_model_tag(model_id)
        strategies = report.get("strategies", {})
        pass_at_k = report.get("pass_at_k", {})
        subset_strategies = report.get("per_subset_strategies", {})
        quality_ablations = report.get("quality_ablations", {})
        case_counts = analysis.get("case_counts", {})
        case_keys.update(case_counts.keys())

        greedy_acc = _safe_get(strategies, "greedy", "accuracy")
        majority_acc = _safe_get(strategies, "majority_vote", "accuracy")
        quality_acc = _safe_get(strategies, "quality", "accuracy")
        a_total_acc = _safe_get(strategies, "a_total", "accuracy")
        a_total_nomode_acc = _safe_get(strategies, "a_total_without_mode", "accuracy")
        q_correct_only_acc = _safe_get(strategies, "quality_only_correct_samples", "accuracy")

        row = {
            "model_id": model_id,
            "method": method,
            "size": size,
            "status": "ok",
            "total": _safe_get(strategies, "greedy", "total", default=0),
            "greedy_acc": greedy_acc,
            "majority_acc": majority_acc,
            "quality_acc": quality_acc,
            "a_total_acc": a_total_acc,
            "a_total_nomode_acc": a_total_nomode_acc,
            "q_correct_only_acc": q_correct_only_acc,
            "delta_a_total_vs_greedy": (a_total_acc - greedy_acc) if isinstance(a_total_acc, (int, float)) and isinstance(greedy_acc, (int, float)) else None,
            "delta_a_total_vs_majority": (a_total_acc - majority_acc) if isinstance(a_total_acc, (int, float)) and isinstance(majority_acc, (int, float)) else None,
            "pass1": _safe_get(pass_at_k, "1", "pass_rate"),
            "pass4": _safe_get(pass_at_k, "4", "pass_rate"),
            "pass8": _safe_get(pass_at_k, "8", "pass_rate"),
            "pass16": _safe_get(pass_at_k, "16", "pass_rate"),
            "partially_correct_total": _safe_get(subset_strategies, "partially_correct", "a_total", "total", default=0),
            "all_correct_total": _safe_get(subset_strategies, "all_correct", "a_total", "total", default=0),
            "all_incorrect_total": _safe_get(subset_strategies, "all_incorrect", "a_total", "total", default=0),
            "empty_extraction_rate": report.get("empty_extraction_rate"),
            "placeholder_extraction_rate": report.get("placeholder_extraction_rate"),
            "nonempty_incorrect_rate": report.get("nonempty_incorrect_rate"),
            "all_correct_group_count": report.get("all_correct_group_count"),
            "avg_all_correct_quality_variance": report.get("avg_all_correct_quality_variance"),
            "ablate_drop_local_check": _safe_get(quality_ablations, "drop_local_check", "accuracy"),
            "ablate_drop_self_verify": _safe_get(quality_ablations, "drop_self_verify", "accuracy"),
            "ablate_format_structure_only": _safe_get(quality_ablations, "format_structure_only", "accuracy"),
            "analysis_total_cases": analysis.get("total_cases", 0),
            "analysis_total_groups": analysis.get("total_groups", 0),
            "train_rollout_step": train_summary.get("rollout_step"),
            "train_optimizer_step": train_summary.get("optimizer_step"),
            "train_group_size": train_summary.get("group_size"),
            "train_lambda_ans": train_summary.get("lambda_ans"),
            "train_lambda_qual": train_summary.get("lambda_qual"),
            "train_lambda_mode": train_summary.get("lambda_mode"),
            "case_counts": case_counts,
            "correct_mode_distribution": report.get("correct_mode_distribution", {}),
            "pass_at_k": pass_at_k,
            "strategies": strategies,
        }
        row_map[model_id] = row

    if expected_models:
        for model_id in expected_models:
            if model_id in row_map:
                continue
            method, size = _parse_model_tag(model_id)
            row_map[model_id] = {
                "model_id": model_id,
                "method": method,
                "size": size,
                "status": "missing_report",
                "total": 0,
                "greedy_acc": None,
                "majority_acc": None,
                "quality_acc": None,
                "a_total_acc": None,
                "a_total_nomode_acc": None,
                "q_correct_only_acc": None,
                "delta_a_total_vs_greedy": None,
                "delta_a_total_vs_majority": None,
                "pass1": None,
                "pass4": None,
                "pass8": None,
                "pass16": None,
                "partially_correct_total": 0,
                "all_correct_total": 0,
                "all_incorrect_total": 0,
                "empty_extraction_rate": None,
                "placeholder_extraction_rate": None,
                "nonempty_incorrect_rate": None,
                "all_correct_group_count": 0,
                "avg_all_correct_quality_variance": None,
                "ablate_drop_local_check": None,
                "ablate_drop_self_verify": None,
                "ablate_format_structure_only": None,
                "analysis_total_cases": 0,
                "analysis_total_groups": 0,
                "train_rollout_step": None,
                "train_optimizer_step": None,
                "train_group_size": None,
                "train_lambda_ans": None,
                "train_lambda_qual": None,
                "train_lambda_mode": None,
                "case_counts": {},
                "correct_mode_distribution": {},
                "pass_at_k": {},
                "strategies": {},
            }

    rows = list(row_map.values())
    rows.sort(key=lambda item: (item.get("a_total_acc") or -1.0), reverse=True)
    return rows, case_keys


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def _to_md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        values = [str(row.get(key, "")) for _, key in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _build_markdown(rows: list[dict[str, Any]], case_keys: set[str]) -> str:
    if not rows:
        return "# math500test Eval Summary\n\nNo *_math500test_report.json files found.\n"

    top = rows[0]
    lines: list[str] = [
        "# math500test Evaluation Summary",
        "",
        f"- Models compared: {len(rows)}",
        f"- Best by a_total: {top['model_id']} ({_pct(top.get('a_total_acc'))})",
        "",
        "## Main Comparison",
        "",
    ]

    main_rows = []
    for row in rows:
        main_rows.append(
            {
                "model": row["model_id"],
                "status": row.get("status", "ok"),
                "method": row["method"],
                "size": row["size"],
                "N": row["total"],
                "greedy": _pct(row.get("greedy_acc")),
                "majority": _pct(row.get("majority_acc")),
                "quality": _pct(row.get("quality_acc")),
                "a_total": _pct(row.get("a_total_acc")),
                "a_total_no_mode": _pct(row.get("a_total_nomode_acc")),
                "q_correct_only": _pct(row.get("q_correct_only_acc")),
                "delta_vs_greedy": _pct(row.get("delta_a_total_vs_greedy")),
                "delta_vs_majority": _pct(row.get("delta_a_total_vs_majority")),
            }
        )
    lines.append(
        _to_md_table(
            main_rows,
            [
                ("Model", "model"),
                ("Status", "status"),
                ("Method", "method"),
                ("Size", "size"),
                ("N", "N"),
                ("Greedy", "greedy"),
                ("Majority", "majority"),
                ("Quality", "quality"),
                ("A_total", "a_total"),
                ("A_total(no mode)", "a_total_no_mode"),
                ("Quality(correct only)", "q_correct_only"),
                ("A_total-Greedy", "delta_vs_greedy"),
                ("A_total-Majority", "delta_vs_majority"),
            ],
        )
    )

    lines.extend(["", "## Pass@k Comparison", ""])
    pass_rows = []
    for row in rows:
        pass_rows.append(
            {
                "model": row["model_id"],
                "pass1": _pct(row.get("pass1")),
                "pass4": _pct(row.get("pass4")),
                "pass8": _pct(row.get("pass8")),
                "pass16": _pct(row.get("pass16")),
            }
        )
    lines.append(
        _to_md_table(
            pass_rows,
            [
                ("Model", "model"),
                ("Pass@1", "pass1"),
                ("Pass@4", "pass4"),
                ("Pass@8", "pass8"),
                ("Pass@16", "pass16"),
            ],
        )
    )

    lines.extend(["", "## Diagnostics", ""])
    diag_rows = []
    for row in rows:
        diag_rows.append(
            {
                "model": row["model_id"],
                "all_correct_groups": row.get("all_correct_group_count", 0),
                "partial_groups": row.get("partially_correct_total", 0),
                "all_incorrect_groups": row.get("all_incorrect_total", 0),
                "empty_extract": _pct(row.get("empty_extraction_rate")),
                "placeholder_extract": _pct(row.get("placeholder_extraction_rate")),
                "nonempty_incorrect": _pct(row.get("nonempty_incorrect_rate")),
                "var_all_correct_quality": _f(row.get("avg_all_correct_quality_variance")),
                "ablate_drop_local": _pct(row.get("ablate_drop_local_check")),
                "ablate_drop_verify": _pct(row.get("ablate_drop_self_verify")),
                "ablate_fmt_struct": _pct(row.get("ablate_format_structure_only")),
            }
        )
    lines.append(
        _to_md_table(
            diag_rows,
            [
                ("Model", "model"),
                ("All-correct groups", "all_correct_groups"),
                ("Partial groups", "partial_groups"),
                ("All-incorrect groups", "all_incorrect_groups"),
                ("Empty extraction", "empty_extract"),
                ("Placeholder extraction", "placeholder_extract"),
                ("Non-empty incorrect", "nonempty_incorrect"),
                ("Var(all-correct quality)", "var_all_correct_quality"),
                ("Drop local_check", "ablate_drop_local"),
                ("Drop self_verify", "ablate_drop_verify"),
                ("Format+Structure only", "ablate_fmt_struct"),
            ],
        )
    )

    if case_keys:
        lines.extend(["", "## Case-Type Counts", ""])
        ordered_case_keys = sorted(case_keys)
        case_rows = []
        for row in rows:
            case_row = {
                "model": row["model_id"],
                "total_cases": row.get("analysis_total_cases", 0),
            }
            for case_key in ordered_case_keys:
                case_row[case_key] = row.get("case_counts", {}).get(case_key, 0)
            case_rows.append(case_row)
        columns = [("Model", "model"), ("Total cases", "total_cases")] + [
            (case_key, case_key) for case_key in ordered_case_keys
        ]
        lines.append(_to_md_table(case_rows, columns))

    lines.extend(["", "## CMAO vs GRPO Delta (Same Size)", ""])
    pair_rows = []
    for size in ("1.5b", "7b"):
        grpo = next((item for item in rows if item["size"] == size and item["method"] == "grpo"), None)
        cmao = next((item for item in rows if item["size"] == size and item["method"] == "cmao"), None)
        if not grpo or not cmao:
            continue
        pair_rows.append(
            {
                "size": size,
                "greedy_delta": _pct((cmao.get("greedy_acc") or 0.0) - (grpo.get("greedy_acc") or 0.0)),
                "majority_delta": _pct((cmao.get("majority_acc") or 0.0) - (grpo.get("majority_acc") or 0.0)),
                "quality_delta": _pct((cmao.get("quality_acc") or 0.0) - (grpo.get("quality_acc") or 0.0)),
                "a_total_delta": _pct((cmao.get("a_total_acc") or 0.0) - (grpo.get("a_total_acc") or 0.0)),
                "pass1_delta": _pct((cmao.get("pass1") or 0.0) - (grpo.get("pass1") or 0.0)),
                "pass16_delta": _pct((cmao.get("pass16") or 0.0) - (grpo.get("pass16") or 0.0)),
            }
        )
    if pair_rows:
        lines.append(
            _to_md_table(
                pair_rows,
                [
                    ("Size", "size"),
                    ("Greedy Δ", "greedy_delta"),
                    ("Majority Δ", "majority_delta"),
                    ("Quality Δ", "quality_delta"),
                    ("A_total Δ", "a_total_delta"),
                    ("Pass@1 Δ", "pass1_delta"),
                    ("Pass@16 Δ", "pass16_delta"),
                ],
            )
        )
    else:
        lines.append("No GRPO/CMAO pair found for same model size.")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate evaluation reports into detailed comparison tables.")
    parser.add_argument("--eval-dir", default="outputs/eval", help="Directory containing report files.")
    parser.add_argument("--train-dir", default="outputs/train", help="Directory containing training summaries.")
    parser.add_argument("--report-suffix", default=REPORT_SUFFIX, help="Suffix used by report files to aggregate.")
    parser.add_argument("--analysis-suffix", default=ANALYSIS_SUFFIX, help="Suffix used by analysis summary files.")
    parser.add_argument(
        "--expected-models",
        default=",".join(EXPECTED_MODELS_DEFAULT),
        help="Comma-separated model ids that must appear in the comparison table.",
    )
    parser.add_argument("--output-prefix", default="outputs/eval/math500test_comparison", help="Output prefix for summary artifacts.")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    train_dir = Path(args.train_dir)
    output_prefix = Path(args.output_prefix)

    expected_models = [item.strip() for item in args.expected_models.split(",") if item.strip()]
    rows, case_keys = _collect_rows(
        eval_dir=eval_dir,
        train_dir=train_dir,
        report_suffix=args.report_suffix,
        analysis_suffix=args.analysis_suffix,
        expected_models=expected_models,
    )
    if not rows:
        raise SystemExit(f"No files matched {eval_dir}/*{args.report_suffix}")

    detailed_columns = [
        "model_id",
        "status",
        "method",
        "size",
        "total",
        "greedy_acc",
        "majority_acc",
        "quality_acc",
        "a_total_acc",
        "a_total_nomode_acc",
        "q_correct_only_acc",
        "delta_a_total_vs_greedy",
        "delta_a_total_vs_majority",
        "pass1",
        "pass4",
        "pass8",
        "pass16",
        "partially_correct_total",
        "all_correct_total",
        "all_incorrect_total",
        "empty_extraction_rate",
        "placeholder_extraction_rate",
        "nonempty_incorrect_rate",
        "all_correct_group_count",
        "avg_all_correct_quality_variance",
        "ablate_drop_local_check",
        "ablate_drop_self_verify",
        "ablate_format_structure_only",
        "analysis_total_cases",
        "analysis_total_groups",
        "train_rollout_step",
        "train_optimizer_step",
        "train_group_size",
        "train_lambda_ans",
        "train_lambda_qual",
        "train_lambda_mode",
    ]
    for key in sorted(case_keys):
        detailed_columns.append(f"case_{key}")

    detailed_rows: list[dict[str, Any]] = []
    for row in rows:
        item = {key: row.get(key) for key in detailed_columns}
        for key in sorted(case_keys):
            item[f"case_{key}"] = row.get("case_counts", {}).get(key, 0)
        detailed_rows.append(item)

    csv_path = output_prefix.with_suffix(".csv")
    _write_csv(csv_path, detailed_rows, detailed_columns)

    strategy_long_path = output_prefix.with_name(output_prefix.name + "_strategy_long.csv")
    strategy_rows: list[dict[str, Any]] = []
    for row in rows:
        for strategy_name, entry in row.get("strategies", {}).items():
            strategy_rows.append(
                {
                    "model_id": row["model_id"],
                    "strategy": strategy_name,
                    "accuracy": entry.get("accuracy"),
                    "correct": entry.get("correct"),
                    "total": entry.get("total"),
                }
            )
    _write_csv(strategy_long_path, strategy_rows, ["model_id", "strategy", "accuracy", "correct", "total"])

    passk_long_path = output_prefix.with_name(output_prefix.name + "_passk_long.csv")
    pass_rows: list[dict[str, Any]] = []
    for row in rows:
        for k, entry in row.get("pass_at_k", {}).items():
            pass_rows.append(
                {
                    "model_id": row["model_id"],
                    "k": int(k),
                    "pass_rate": entry.get("pass_rate"),
                    "correct": entry.get("correct"),
                    "total": entry.get("total"),
                }
            )
    pass_rows.sort(key=lambda item: (item["model_id"], item["k"]))
    _write_csv(passk_long_path, pass_rows, ["model_id", "k", "pass_rate", "correct", "total"])

    summary_json_path = output_prefix.with_suffix(".json")
    summary_payload = {
        "models": detailed_rows,
        "generated_from": str(eval_dir),
        "report_count": len(rows),
    }
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

    md_path = output_prefix.with_suffix(".md")
    md_text = _build_markdown(rows=rows, case_keys=case_keys)
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(md_text)

    print("[DONE] math500test comparison artifacts generated:")
    print(f"- {csv_path}")
    print(f"- {strategy_long_path}")
    print(f"- {passk_long_path}")
    print(f"- {summary_json_path}")
    print(f"- {md_path}")

    print("\n[TOP by a_total]")
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank}. {row['model_id']} | a_total={_pct(row.get('a_total_acc'))} | "
            f"greedy={_pct(row.get('greedy_acc'))} | majority={_pct(row.get('majority_acc'))} | pass@16={_pct(row.get('pass16'))}"
        )


if __name__ == "__main__":
    main()
