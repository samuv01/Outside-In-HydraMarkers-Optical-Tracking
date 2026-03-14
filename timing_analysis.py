# This script analyses a timing summary CSV produced by ReadMarker_FromNPZ.py
# It generates plots of mean times and stds per task and prints a summary to the console.

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TimingStats:
    stage: str
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    count: int


def load_timing_summary(csv_path: Path) -> List[TimingStats]:
    timings: List[TimingStats] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"stage", "mean_ms", "median_ms", "std_ms", "min_ms", "max_ms", "count"}
        if not required.issubset(reader.fieldnames or set()):
            missing = required.difference(set(reader.fieldnames or []))
            raise ValueError(f"Timing summary is missing columns: {sorted(missing)}")
        for row in reader:
            timings.append(
                TimingStats(
                    stage=row["stage"],
                    mean_ms=float(row["mean_ms"]),
                    median_ms=float(row["median_ms"]),
                    std_ms=float(row["std_ms"]),
                    min_ms=float(row["min_ms"]),
                    max_ms=float(row["max_ms"]),
                    count=int(float(row["count"])),
                )
            )
    if not timings:
        raise ValueError(f"No timing data found in {csv_path}")
    return timings


def plot_timings(timings: List[TimingStats], *, output_dir: Path) -> None:
    filtered = [stat for stat in timings if stat.stage.lower() != "total"]
    if not filtered:
        raise ValueError("Timing summary only contains the 'total' stage; per-stage analysis unavailable.")

    stages = [stat.stage for stat in filtered]
    means = np.array([stat.mean_ms for stat in filtered])
    stds = np.array([stat.std_ms for stat in filtered])

    total_mean = means.sum()
    percentages = means / total_mean * 100.0 if total_mean > 0 else np.zeros_like(means)

    colors = plt.cm.tab20(np.linspace(0, 1, len(stages)))
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_positions = np.arange(len(stages))
    bars = ax.bar(bar_positions, means, yerr=stds, capsize=6, color=colors[: len(stages)])
    ax.set_xticks(bar_positions, stages, rotation=30, ha='right')
    ax.set_ylabel('Mean time (ms)')
    ax.set_title('Mean Time per Task')
    for rect, pct in zip(bars, percentages):
        height = rect.get_height()
        ax.annotate(
            f"{pct:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(10, 0),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'timing_means.png').write_bytes(_figure_to_png(fig))

    print("\nTiming summary (mean ± std in ms):")
    for stat, pct in zip(timings, percentages):
        print(
            f"  {stat.stage:>20}: {stat.mean_ms:8.3f} ± {stat.std_ms:6.3f} ms "
            f"({pct:5.1f}% of total, n={stat.count})"
        )


def _figure_to_png(fig: plt.Figure) -> bytes:
    import io

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse timing summary CSV and generate visualisations."
    )
    parser.add_argument(
        "summary_csv",
        nargs="?",
        type=Path,
        help="Path to timings_summary.csv produced by ReadMarker_FromNPZ.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where plots will be saved (defaults to alongside the CSV).",
    )
    return parser.parse_args()


def _prompt_for_csv(default_path: Path) -> Optional[Path]:
    try:
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
    except Exception:
        return None

    root = Tk()
    root.withdraw()
    file_path = askopenfilename(
        title="Select timings_summary.csv",
        initialdir=str(default_path),
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    if not file_path:
        return None
    return Path(file_path)


def main() -> None:
    args = parse_args()
    summary_csv = args.summary_csv

    if summary_csv is None:
        summary_csv = _prompt_for_csv(default_path=Path.cwd())
        if summary_csv is None:
            print("No timing summary selected; exiting.")
            return
    summary_csv = summary_csv.resolve()

    if args.output_dir is None:
        output_dir = summary_csv.parent
    else:
        output_dir = args.output_dir.resolve()

    if not summary_csv.exists():
        raise FileNotFoundError(f"Timing summary not found: {summary_csv}")
    stats = load_timing_summary(summary_csv)
    plot_timings(stats, output_dir=output_dir)


if __name__ == "__main__":
    main()
