"""Run directories and metric logging (stdout + CSV + TensorBoard)."""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str = "auto") -> torch.device:
    """``auto`` picks CUDA, then Apple-silicon MPS, then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def make_run_dir(cfg: dict[str, Any], root: str | Path = "runs") -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(root) / f"{cfg['name']}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return run_dir


class Logger:
    """Writes scalar metrics to ``metrics.csv`` and TensorBoard, and prints a summary line."""

    def __init__(self, run_dir: str | Path, use_tensorboard: bool = True):
        self.run_dir = Path(run_dir)
        self.start = time.time()
        self._csv_file = open(self.run_dir / "metrics.csv", "w", newline="")
        self._csv_writer: csv.DictWriter | None = None
        self._fields: list[str] = []
        self._tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(str(self.run_dir / "tb"))
            except Exception:  # tensorboard is optional
                self._tb = None

    def log(self, step: int, metrics: dict[str, float], echo: bool = True) -> None:
        row = {"step": step, "time": round(time.time() - self.start, 1), **metrics}
        if self._csv_writer is None or any(k not in self._fields for k in row):
            # (Re)open the header when new metric names appear.
            self._fields = list(dict.fromkeys([*self._fields, *row]))
            self._csv_file.close()
            existing = []
            path = self.run_dir / "metrics.csv"
            if path.exists() and path.stat().st_size:
                with open(path) as f:
                    existing = list(csv.DictReader(f))
            self._csv_file = open(path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fields)
            self._csv_writer.writeheader()
            self._csv_writer.writerows(existing)
        self._csv_writer.writerow(row)
        self._csv_file.flush()
        if self._tb is not None:
            for k, v in metrics.items():
                if v is not None and np.isfinite(v):
                    self._tb.add_scalar(k, v, step)
        if echo:
            parts = [f"{k}={_fmt(v)}" for k, v in row.items()]
            print(" | ".join(parts), flush=True)

    def write_summary(self, summary: dict[str, Any]) -> None:
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def close(self) -> None:
        self._csv_file.close()
        if self._tb is not None:
            self._tb.close()


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)
