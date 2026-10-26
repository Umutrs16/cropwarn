"""Harvested-area states and production-interval error propagation."""

from __future__ import annotations

import numpy as np
import pandas as pd


AREA_STATES = ("active", "damaged", "fallow", "unharvested")


def classify_area_state(row: pd.Series) -> str:
    shock = float(row.get("shock_true", 0.0))
    if shock >= 0.20:
        return "damaged"
    if shock >= 0.12:
        return "unharvested"
    return "active"


def production_interval(
    yield_pred: np.ndarray,
    yield_width: np.ndarray,
    area_ha: np.ndarray,
    area_cv: float = 0.06,
    yield_cv: float = 0.08,
    draws: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(yield_pred)
    y_sd = np.clip(yield_width / 3.29, yield_pred * yield_cv, None)
    a_sd = np.clip(area_ha * area_cv, 1.0, None)
    samples = []
    for _ in range(draws):
        y = rng.normal(yield_pred, y_sd)
        a = rng.normal(area_ha, a_sd)
        samples.append(np.clip(y, 0.05, None) * np.clip(a, 1.0, None))
    sim = np.vstack(samples)
    return pd.DataFrame(
        {
            "production_p10": np.quantile(sim, 0.10, axis=0),
            "production_p50": np.quantile(sim, 0.50, axis=0),
            "production_p90": np.quantile(sim, 0.90, axis=0),
        }
    )


def attach_area_states(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["area_state"] = out.apply(classify_area_state, axis=1)
    return out
