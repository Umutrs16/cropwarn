"""Configuration loading and path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    root: Path

    def get(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    def path(self, key: str) -> Path:
        rel = self.get("paths", key, default=key)
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    @property
    def seed(self) -> int:
        return int(self.get("project", "random_seed", default=42))

    @property
    def crops(self) -> list[str]:
        return list(self.get("study", "crops"))

    @property
    def years(self) -> list[int]:
        start = int(self.get("study", "year_start"))
        end = int(self.get("study", "year_end"))
        return list(range(start, end + 1))

    def leads(self, demo: bool = False) -> list[int]:
        key = "demo_leads_days" if demo else "leads_days"
        return [int(x) for x in self.get("study", key)]


def load_config(root: str | Path | None = None, extra: dict[str, Any] | None = None) -> Config:
    root_path = Path(root) if root else Path(__file__).resolve().parents[2]
    merged: dict[str, Any] = {}
    for name in ("default.yaml", "experiments.yaml", "data.yaml"):
        path = root_path / "configs" / name
        with path.open(encoding="utf-8") as f:
            merged = _deep_merge(merged, yaml.safe_load(f) or {})
    if extra:
        merged = _deep_merge(merged, extra)
    return Config(raw=merged, root=root_path)


def ensure_dirs(cfg: Config) -> None:
    for key in ("data_raw", "data_interim", "features", "models", "results", "figures"):
        cfg.path(key).mkdir(parents=True, exist_ok=True)
