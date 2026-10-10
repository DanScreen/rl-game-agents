"""Learning curves from one or more runs' ``metrics.csv`` files."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


def read_metric(run_dir: Path, metric: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    with open(run_dir / "metrics.csv") as f:
        for row in csv.DictReader(f):
            v = row.get(metric)
            if v not in (None, "", "nan"):
                y = float(v)
                if math.isfinite(y):
                    xs.append(float(row["step"]))
                    ys.append(y)
    return xs, ys


def default_metric(run_dir: Path) -> str:
    with open(run_dir / "metrics.csv") as f:
        header = next(csv.reader(f))
    for m in ("eval/mean_return", "eval/mean_score"):
        if m in header:
            return m
    return header[2]


def plot_runs(run_dirs: list[Path], out: Path, metrics: list[str] | None = None, title: str | None = None) -> Path:
    metrics = metrics or [default_metric(run_dirs[0])]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for run in run_dirs:
            xs, ys = read_metric(run, metric)
            if xs:
                ax.plot(xs, ys, marker="o" if len(xs) < 40 else None, markersize=3, label=run.name)
        ax.set_title(metric)
        ax.set_xlabel("games" if "score" in metric else "environment steps")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
