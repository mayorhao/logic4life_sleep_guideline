"""Validated experiment settings loaded from JSON and command-line overrides."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class Config:
    generation_model: str = "gpt-5-mini-2025-08-07"
    embedding_model: str = "text-embedding-3-small"
    runs: int = 40
    concurrency: int = 16
    embedding_batch_size: int = 128
    max_attempts: int = 5
    bootstrap_samples: int = 5000
    confidence: float = 0.95
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("runs", "concurrency", "embedding_batch_size",
                     "max_attempts", "bootstrap_samples"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if (type(self.confidence) not in (int, float)
                or not math.isfinite(self.confidence) or not 0 < self.confidence < 1):
            raise ValueError("confidence must be a finite number between 0 and 1")
        for name in ("generation_model", "embedding_model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")


def load_config(path: Path | None = None,
                overrides: Mapping[str, Any] | None = None) -> Config:
    """Apply defaults, then file values, then explicitly supplied overrides."""
    values: dict[str, Any] = {}
    valid_keys = {field.name for field in fields(Config)}
    if path is not None:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = set(data) - valid_keys
        if unknown:
            raise ValueError(f"Unknown configuration fields: {', '.join(sorted(unknown))}")
        # Reject invalid file values even when an override would replace them.
        Config(**data)
        values.update(data)
    if overrides:
        unknown = set(overrides) - valid_keys
        if unknown:
            raise ValueError(f"Unknown configuration overrides: {', '.join(sorted(unknown))}")
        values.update(overrides)
    return Config(**values)
