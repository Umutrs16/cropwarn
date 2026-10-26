"""Country–crop–year cluster bootstrap for R² and isolated lead."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from cropwarn.validation.metrics import earliest_reliable_lead, stable_reliable_lead


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 8:
        return float("nan")
    return float(r2_score(y[m], p[m]))


def cluster_bootstrap_lead(
    preds: pd.DataFrame,
    experiments: tuple[str, ...] = ("B2_optical_climate", "B4_plus_sif"),
    n_boot: int = 400,
    seed: int = 42,
    r2_floor: float = 0.15,
) -> pd.DataFrame:
    keys = [c for c in ("iso3", "crop", "year") if c in preds.columns]
    p = preds[preds["experiment"].isin(experiments)].copy()
    if p.empty or "pred" not in p.columns:
        return pd.DataFrame()
    ycol = "yield_anomaly" if "yield_anomaly" in p.columns else "y"
    clusters = p[keys].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_boot):
        idx = rng.integers(0, len(clusters), size=len(clusters))
        draw = clusters.iloc[idx].copy()
        draw["_boot"] = np.arange(len(draw))
        sample = p.merge(draw, on=keys, how="inner")
        for exp, g in sample.groupby("experiment"):
            skill_rows = []
            for lead, gg in g.groupby("lead"):
                skill_rows.append({"lead": int(lead), "r2": _r2(gg[ycol].to_numpy(), gg["pred"].to_numpy())})
            skill = pd.DataFrame(skill_rows)
            rows.append(
                {
                    "boot": b,
                    "experiment": exp,
                    "isolated_lead": earliest_reliable_lead(skill, r2_floor=r2_floor),
                    "stable_lead": stable_reliable_lead(skill, r2_floor=r2_floor),
                    **{f"r2_{int(r.lead)}": r.r2 for r in skill.itertuples()},
                }
            )
    boots = pd.DataFrame(rows)
    out = []
    for exp, g in boots.groupby("experiment"):
        rec = {"experiment": exp, "n_boot": int(len(g))}
        for col in ("isolated_lead", "stable_lead"):
            s = pd.to_numeric(g[col], errors="coerce")
            rec[f"{col}_p50"] = float(s.median())
            rec[f"{col}_p05"] = float(s.quantile(0.05))
            rec[f"{col}_p95"] = float(s.quantile(0.95))
        for col in [c for c in g.columns if c.startswith("r2_")]:
            s = pd.to_numeric(g[col], errors="coerce")
            rec[f"{col}_p50"] = float(s.median())
            rec[f"{col}_p05"] = float(s.quantile(0.05))
            rec[f"{col}_p95"] = float(s.quantile(0.95))
        out.append(rec)
    # ΔR² B4 − B2 at each lead
    if {"B2_optical_climate", "B4_plus_sif"} <= set(boots["experiment"]):
        b2 = boots[boots["experiment"] == "B2_optical_climate"]
        b4 = boots[boots["experiment"] == "B4_plus_sif"]
        merged = b2.merge(b4, on="boot", suffixes=("_b2", "_b4"))
        rec = {"experiment": "B4_minus_B2", "n_boot": int(len(merged))}
        for lead in (0, 32, 64, 96, 128, 160, 192):
            col = f"r2_{lead}"
            if f"{col}_b2" not in merged.columns:
                continue
            delta = pd.to_numeric(merged[f"{col}_b4"], errors="coerce") - pd.to_numeric(
                merged[f"{col}_b2"], errors="coerce"
            )
            rec[f"d{col}_p50"] = float(delta.median())
            rec[f"d{col}_p05"] = float(delta.quantile(0.05))
            rec[f"d{col}_p95"] = float(delta.quantile(0.95))
            rec[f"d{col}_gt0"] = float((delta > 0).mean())
        lead_d = pd.to_numeric(merged["isolated_lead_b4"], errors="coerce") - pd.to_numeric(
            merged["isolated_lead_b2"], errors="coerce"
        )
        rec["disolated_p50"] = float(lead_d.median())
        rec["disolated_p05"] = float(lead_d.quantile(0.05))
        rec["disolated_p95"] = float(lead_d.quantile(0.95))
        rec["disolated_ge14"] = float((lead_d >= 14).mean())
        stable_d = pd.to_numeric(merged["stable_lead_b4"], errors="coerce") - pd.to_numeric(
            merged["stable_lead_b2"], errors="coerce"
        )
        rec["dstable_p50"] = float(stable_d.median())
        rec["dstable_p05"] = float(stable_d.quantile(0.05))
        rec["dstable_p95"] = float(stable_d.quantile(0.95))
        rec["dstable_ge14"] = float((stable_d >= 14).mean())
        rec["dstable_ge96"] = float((stable_d >= 96).mean())
        out.append(rec)
    return pd.DataFrame(out)
