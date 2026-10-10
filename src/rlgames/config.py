"""YAML experiment configs with `key=value` command-line overrides.

A config is a flat-ish YAML mapping. Nested keys are addressed with dots on the
command line, e.g. ``rlg train configs/cartpole_dqn.yaml total_steps=20000 algo.lr=1e-3``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with open(path) as f:
        cfg = _coerce_numbers(yaml.safe_load(f) or {})
    cfg = apply_overrides(cfg, overrides or [])
    cfg.setdefault("name", Path(path).stem)
    return cfg


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        value = _coerce_numbers(yaml.safe_load(raw))  # parses ints, floats, bools, lists
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


def _coerce_numbers(value: Any) -> Any:
    """YAML 1.1 reads ``3e-4`` as a string; turn such strings into floats."""
    if isinstance(value, dict):
        return {k: _coerce_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_numbers(v) for v in value]
    if isinstance(value, str) and any(c.isdigit() for c in value):  # leaves "nan"/"inf" names alone
        try:
            return float(value)
        except ValueError:
            return value
    return value
