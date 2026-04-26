from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PANELS = (
    ("reward", ("weighted_reward_mean", "a_total_abs_mean", "a_ans_mean", "a_qual_mean", "a_mode_mean")),
    ("optimization", ("loss", "policy_loss", "kl", "clip_fraction")),
    ("rollout", ("correct_ratio", "nonzero_advantage_ratio", "truncated_completion_ratio")),
)


def load_metrics(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    if not records:
        raise ValueError(f"No metrics found in {path}")
    return records


def numeric_series(records: list[dict[str, Any]], key: str) -> list[float] | None:
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if not isinstance(value, (int, float)):
            return None
        values.append(float(value))
    return values


def plot_metrics(
    metrics_path: str | Path,
    output_path: str | Path,
    *,
    x_key: str = "iteration",
    panels: tuple[tuple[str, tuple[str, ...]], ...] = DEFAULT_PANELS,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('Plotting requires matplotlib. Install it with: pip install -e ".[plot]"') from exc

    records = load_metrics(metrics_path)
    x_values = numeric_series(records, x_key)
    if x_values is None:
        x_values = [float(index + 1) for index in range(len(records))]

    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3.2 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]

    for axis, (title, metric_names) in zip(axes, panels):
        plotted = False
        for metric_name in metric_names:
            values = numeric_series(records, metric_name)
            if values is None:
                continue
            axis.plot(x_values, values, marker="o", markersize=2.5, linewidth=1.4, label=metric_name)
            plotted = True
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        if plotted:
            axis.legend(loc="best", fontsize=8)
        else:
            axis.text(0.5, 0.5, "no available metrics", ha="center", va="center", transform=axis.transAxes)

    axes[-1].set_xlabel(x_key if x_values is not None else "step")
    fig.tight_layout()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot online GRPO/CMAO training metrics.")
    parser.add_argument("--input", required=True, help="Path to online_metrics.jsonl.")
    parser.add_argument("--output", required=True, help="Output image path, usually .png.")
    parser.add_argument("--x-key", default="iteration", help="Metric key to use as x-axis.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_metrics(args.input, args.output, x_key=args.x_key)
    print(f"Saved training metric plot to {args.output}")


if __name__ == "__main__":
    main()
