from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_METRICS = (
    "reward",
    "reward_std",
    "rewards/CMAORewardFunction/mean",
    "rewards/CMAORewardFunction/std",
    "frac_reward_zero_std",
    "completions/mean_length",
    "completions/min_length",
    "completions/max_length",
    "completions/clipped_ratio",
    "kl",
    "loss",
    "grad_norm",
    "learning_rate",
    "clip_ratio/low_mean",
    "clip_ratio/high_mean",
    "clip_ratio/region_mean",
    "correct_ratio",
    "correct_count",
    "policy_loss",
    "clip_fraction",
    "weighted_reward_mean",
    "weighted_reward_std",
    "a_total_abs_mean",
    "a_ans_mean",
    "a_qual_mean",
    "a_mode_mean",
    "nonzero_advantage_ratio",
    "zero_advantage_group_count",
    "truncated_completion_ratio",
    "response_tokens_mean",
    "sample_count",
    "optimizer_step",
)

METRIC_LABELS = {
    "step": "Step",
    "iteration": "Iteration",
    "reward": "Reward",
    "reward_std": "Reward Std",
    "rewards/CMAORewardFunction/mean": "CMAO Reward Mean",
    "rewards/CMAORewardFunction/std": "CMAO Reward Std",
    "frac_reward_zero_std": "Zero-Std Reward Fraction",
    "completions/mean_length": "Completion Mean Length",
    "completions/min_length": "Completion Min Length",
    "completions/max_length": "Completion Max Length",
    "completions/clipped_ratio": "Completion Clipped Ratio",
    "grad_norm": "Gradient Norm",
    "learning_rate": "Learning Rate",
    "clip_ratio/low_mean": "Low Clip Ratio",
    "clip_ratio/high_mean": "High Clip Ratio",
    "clip_ratio/region_mean": "Clip Region Ratio",
    "correct_ratio": "Correct Ratio",
    "correct_count": "Correct Count",
    "loss": "Loss",
    "policy_loss": "Policy Loss",
    "kl": "KL",
    "clip_fraction": "Clip Fraction",
    "weighted_reward_mean": "Weighted Reward Mean",
    "weighted_reward_std": "Weighted Reward Std",
    "a_total_abs_mean": "|Total Advantage| Mean",
    "a_ans_mean": "Answer Advantage Mean",
    "a_qual_mean": "Quality Advantage Mean",
    "a_mode_mean": "Mode Advantage Mean",
    "nonzero_advantage_ratio": "Nonzero Advantage Ratio",
    "zero_advantage_group_count": "Zero-Advantage Groups",
    "truncated_completion_ratio": "Truncated Completion Ratio",
    "response_tokens_mean": "Response Tokens Mean",
    "sample_count": "Sample Count",
    "optimizer_step": "Optimizer Step",
}


@dataclass(frozen=True)
class RunMetrics:
    label: str
    path: Path
    records: list[dict[str, Any]]


def load_metrics(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    stripped_text = text.strip()
    if not stripped_text:
        raise ValueError(f"No metrics found in {path}")

    whole_document = _load_whole_json_document(stripped_text)
    if whole_document is not None:
        records = _records_from_json_document(whole_document)
        if records:
            return normalize_records(records)

    records: list[dict[str, Any]] = []
    embedded_log_history: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for line in text.splitlines():
        record = _decode_json_object_from_line(line, decoder)
        if record is not None:
            if "log_history" in record and isinstance(record["log_history"], list):
                embedded_log_history.extend(item for item in record["log_history"] if isinstance(item, dict))
            elif _looks_like_metric_record(record):
                records.append(record)
    if not records and embedded_log_history:
        records = embedded_log_history
    if not records:
        raise ValueError(f"No metrics found in {path}")
    return normalize_records(records)


def _load_whole_json_document(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _records_from_json_document(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, list):
        return [item for item in document if isinstance(item, dict)]
    if isinstance(document, dict):
        log_history = document.get("log_history")
        if isinstance(log_history, list):
            return [item for item in log_history if isinstance(item, dict)]
        if _looks_like_metric_record(document):
            return [document]
    return []


def _decode_json_object_from_line(line: str, decoder: json.JSONDecoder) -> dict[str, Any] | None:
    start = line.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(line[start:])
        except json.JSONDecodeError:
            start = line.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = line.find("{", start + 1)
    return None


def _looks_like_metric_record(record: dict[str, Any]) -> bool:
    return any(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in record.values())


def normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        item = dict(record)
        if not isinstance(item.get("step"), (int, float)):
            parsed_step = _parse_global_step(item.get("global_step/max_steps"))
            if parsed_step is not None:
                item["step"] = parsed_step
        if not isinstance(item.get("iteration"), (int, float)):
            item["iteration"] = item["step"] if isinstance(item.get("step"), (int, float)) else index
        normalized.append(item)
    return normalized


def _parse_global_step(value: Any) -> int | None:
    if isinstance(value, str):
        match = re.match(r"\s*(\d+)\s*/\s*\d+\s*$", value)
        if match:
            return int(match.group(1))
    return None


def numeric_series(records: list[dict[str, Any]], key: str) -> list[float] | None:
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return None
        values.append(float(value))
    return values


def metric_xy_series(records: list[dict[str, Any]], metric_key: str, x_key: str) -> tuple[list[float], list[float]]:
    pairs_by_x: dict[float, float] = {}
    fallback_index = 1
    for record in records:
        y_value = record.get(metric_key)
        if not isinstance(y_value, (int, float)) or not math.isfinite(float(y_value)):
            fallback_index += 1
            continue
        x_value = record.get(x_key)
        if isinstance(x_value, (int, float)) and math.isfinite(float(x_value)):
            x_coord = float(x_value)
        else:
            x_coord = float(fallback_index)
        pairs_by_x[x_coord] = float(y_value)
        fallback_index += 1
    pairs = sorted(pairs_by_x.items())
    return [x_value for x_value, _ in pairs], [y_value for _, y_value in pairs]


def available_numeric_metrics(runs: Iterable[RunMetrics], x_key: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for run in runs:
        for record in run.records:
            for key, value in record.items():
                if key == x_key or key in seen:
                    continue
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    ordered.append(key)
                    seen.add(key)
    preferred = [metric for metric in DEFAULT_METRICS if metric in seen]
    extras = sorted(metric for metric in ordered if metric not in DEFAULT_METRICS)
    return preferred + extras


def smooth_series(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < 3:
        return values
    safe_window = max(1, min(int(window), len(values)))
    if safe_window % 2 == 0:
        safe_window -= 1
    radius = safe_window // 2
    smoothed: list[float] = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def infer_label(path: Path) -> str:
    parent = path.parent.name
    if parent and parent != ".":
        return parent
    return path.stem


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip().lower())
    return slug.strip("_") or "metric"


def output_path_for_metric(output_dir: Path, metric_name: str, file_format: str) -> Path:
    return output_dir / f"{slugify(metric_name)}.{file_format}"


def apply_paper_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (6.0, 3.6),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_metric(
    runs: list[RunMetrics],
    metric_name: str,
    output_path: str | Path,
    *,
    x_key: str = "iteration",
    smooth: int = 7,
    title: str | None = None,
) -> bool:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('Plotting requires matplotlib. Install it with: pip install -e ".[plot]"') from exc

    apply_paper_style(plt)
    fig, axis = plt.subplots()
    colors = plt.get_cmap("tab10").colors
    plotted = False

    for run_index, run in enumerate(runs):
        x_values, y_values = metric_xy_series(run.records, metric_name, x_key)
        if not y_values:
            continue
        y_values = smooth_series(y_values, smooth)
        axis.plot(
            x_values,
            y_values,
            label=run.label,
            color=colors[run_index % len(colors)],
            alpha=0.95,
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    display_name = METRIC_LABELS.get(metric_name, metric_name.replace("_", " ").title())
    axis.set_title(title or display_name, pad=8)
    axis.set_xlabel(METRIC_LABELS.get(x_key, x_key.replace("_", " ").title()))
    axis.set_ylabel(display_name)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
    axis.grid(True, axis="y", color="#D9DEE7", linewidth=0.75, alpha=0.75)
    axis.grid(True, axis="x", color="#EEF1F5", linewidth=0.55, alpha=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="best")
    fig.tight_layout(pad=0.8)

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_metrics(
    metrics_paths: list[str | Path],
    output_dir: str | Path,
    *,
    labels: list[str] | None = None,
    metrics: list[str] | None = None,
    x_key: str = "iteration",
    smooth: int = 7,
    file_format: str = "png",
) -> list[Path]:
    paths = [Path(path) for path in metrics_paths]
    if labels is not None and len(labels) != len(paths):
        raise ValueError("--labels must have the same number of entries as --inputs")
    runs = [
        RunMetrics(
            label=labels[index] if labels is not None else infer_label(path),
            path=path,
            records=load_metrics(path),
        )
        for index, path in enumerate(paths)
    ]
    selected_metrics = metrics or available_numeric_metrics(runs, x_key=x_key)
    target_dir = Path(output_dir)
    saved: list[Path] = []
    for metric_name in selected_metrics:
        target = output_path_for_metric(target_dir, metric_name, file_format)
        if plot_metric(runs, metric_name, target, x_key=x_key, smooth=smooth):
            saved.append(target)
    if not saved:
        raise ValueError("No requested numeric metrics were available to plot.")
    return saved


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    if args.output:
        output = Path(args.output)
        if output.suffix:
            return output.with_suffix("")
        return output
    raise ValueError("Provide --output-dir for multi-metric plots, or --output for backward compatibility.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot clean per-metric GRPO/CMAO training curves from online metrics, "
            "Swift nohup logs, or trainer_state/log_history JSON."
        )
    )
    parser.add_argument("--input", help="Single metrics/log path.")
    parser.add_argument("--inputs", nargs="+", help="One or more metrics/log paths to compare.")
    parser.add_argument("--labels", nargs="+", help="Labels for each input run.")
    parser.add_argument("--output", help="Backward-compatible output path. Its stem is used as an output directory.")
    parser.add_argument("--output-dir", help="Directory where one image per metric will be saved.")
    parser.add_argument("--metrics", nargs="+", help="Metric keys to plot. Defaults to all known numeric metrics.")
    parser.add_argument("--x-key", default="iteration", help="Metric key to use as x-axis.")
    parser.add_argument("--smooth", type=int, default=7, help="Centered moving-average window. Use 1 to disable.")
    parser.add_argument("--format", default="png", choices=("png", "pdf", "svg"), help="Output image format.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_paths = args.inputs or ([args.input] if args.input else None)
    if not input_paths:
        raise SystemExit("Provide --inputs or --input.")
    output_dir = resolve_output_dir(args)
    saved = plot_metrics(
        input_paths,
        output_dir,
        labels=args.labels,
        metrics=args.metrics,
        x_key=args.x_key,
        smooth=args.smooth,
        file_format=args.format,
    )
    print(f"Saved {len(saved)} training metric plots to {output_dir}")


if __name__ == "__main__":
    main()
