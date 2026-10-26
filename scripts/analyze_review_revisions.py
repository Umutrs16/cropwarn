"""Post-hoc revision metrics from official wheat+soy caches. Does not retrain."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.data.io import read_table, write_table
from cropwarn.trade.exposure import (
    evaluate_realized_import_shortfall,
    evaluate_trade,
    trade_early_warning_skill,
)
from cropwarn.validation.metrics import earliest_reliable_lead, skill_score, stable_reliable_lead

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision_p0"


def _stable_table(lead_skill: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exp, g in lead_skill.groupby("experiment"):
        rows.append(
            {
                "experiment": exp,
                "isolated_earliest_lead": earliest_reliable_lead(g),
                "stable_lead": stable_reliable_lead(g),
                "harvest_r2": float(g.loc[g["lead"] == 0, "r2"].mean()) if (g["lead"] == 0).any() else np.nan,
                "r2_192": float(g.loc[g["lead"] == 192, "r2"].mean()) if (g["lead"] == 192).any() else np.nan,
                "min_r2_to_stable": float(
                    g.loc[g["lead"] <= (stable_reliable_lead(g) if np.isfinite(stable_reliable_lead(g)) else -1), "r2"].min()
                )
                if np.isfinite(stable_reliable_lead(g))
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _skill_vs_baselines(preds: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("iso3", "unit_id", "crop", "year") if c in preds.columns and c in panel.columns]
    p = preds[preds["experiment"] == "B2_optical_climate"].copy()
    if p.empty or "yield_anomaly" not in p.columns:
        return pd.DataFrame()
    prev = panel[keys + ["yield_anomaly"]].drop_duplicates().copy()
    prev["year"] = prev["year"] + 1
    prev = prev.rename(columns={"yield_anomaly": "prev_anom"})
    p = p.merge(prev, on=keys, how="left")
    rows = []
    for lead, g in p.groupby("lead"):
        y = pd.to_numeric(g["yield_anomaly"], errors="coerce")
        hat = pd.to_numeric(g["pred"], errors="coerce")
        m = y.notna() & hat.notna()
        yv, hv = y[m].to_numpy(), hat[m].to_numpy()
        mse = float(np.mean((yv - hv) ** 2)) if len(yv) else np.nan
        mse0 = float(np.mean(yv**2)) if len(yv) else np.nan
        persist = np.nan
        both = g["yield_anomaly"].notna() & g["prev_anom"].notna()
        if both.any():
            persist = float(np.mean((g.loc[both, "yield_anomaly"] - g.loc[both, "prev_anom"]) ** 2))
        rows.append(
            {
                "lead": int(lead),
                "mse_b2": mse,
                "ss_vs_zero": skill_score(mse, mse0),
                "ss_vs_persistence": skill_score(mse, persist),
                "n": int(m.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("lead", ascending=False)


def _sample_sizes(preds: pd.DataFrame, panel: pd.DataFrame) -> dict:
    p = preds[preds["experiment"].isin(["B2_optical_climate", "B4_plus_sif"])].copy()
    keys = [c for c in ("iso3", "unit_id", "crop", "year") if c in p.columns]
    years = {2010, 2012, 2015, 2016, 2019, 2020, 2022, 2003}
    if "yield_z" in panel.columns:
        z = panel[[c for c in keys + ["yield_z"] if c in panel.columns or c == "yield_z"]].drop_duplicates()
        overlap = [c for c in keys if c in z.columns]
        p = p.merge(z, on=overlap, how="left") if overlap else p
        shock = p["year"].isin(years) | (pd.to_numeric(p.get("yield_z"), errors="coerce").abs() >= 1)
    else:
        shock = p["year"].isin(years)
    s = p[shock]
    return {
        "official_prediction_rows": int(len(preds)),
        "b2_b4_rows": int(len(p)),
        "shock_prediction_rows": int(len(s)),
        "shock_country_crop_years": int(s[keys].drop_duplicates().shape[0]) if keys else 0,
        "shock_countries": int(s["iso3"].nunique()) if "iso3" in s.columns else 0,
        "headline_country_crop_years": int(p[keys].drop_duplicates().shape[0]) if keys else 0,
        "headline_countries": int(p["iso3"].nunique()) if "iso3" in p.columns else 0,
        "headline_crops": sorted(p["crop"].dropna().astype(str).unique().tolist()) if "crop" in p.columns else [],
        "test_years": sorted(int(y) for y in p["year"].dropna().unique()) if "year" in p.columns else [],
    }


def _coverage_slices(preds: pd.DataFrame) -> pd.DataFrame:
    full = preds[preds["experiment"].isin(["P_full_physiology", "P_food_system"])].copy()
    if full.empty or not {"lower", "upper", "yield_anomaly"} <= set(full.columns):
        return pd.DataFrame()
    y = pd.to_numeric(full["yield_anomaly"], errors="coerce")
    lo = pd.to_numeric(full["lower"], errors="coerce")
    hi = pd.to_numeric(full["upper"], errors="coerce")
    full["covered"] = (y >= lo) & (y <= hi)
    rows = []
    for dim in ("crop", "region", "year"):
        if dim not in full.columns:
            continue
        for key, g in full.groupby(dim):
            rows.append(
                {
                    "slice": dim,
                    "key": str(key),
                    "coverage90": float(g["covered"].mean()),
                    "n": int(len(g)),
                    "mae": float(pd.to_numeric(g.get("abs_err"), errors="coerce").mean()),
                }
            )
    return pd.DataFrame(rows)


def _selective(preds: pd.DataFrame) -> pd.DataFrame:
    full = preds[preds["experiment"] == "P_full_physiology"].copy()
    if full.empty or "ood_score" not in full.columns:
        return pd.DataFrame()
    err = pd.to_numeric(full.get("abs_err"), errors="coerce")
    ood = pd.to_numeric(full["ood_score"], errors="coerce")
    m = err.notna() & ood.notna()
    err, ood = err[m].to_numpy(), ood[m].to_numpy()
    order = np.argsort(ood)
    err = err[order]
    rows = []
    for reject in (0.0, 0.05, 0.10, 0.20, 0.30):
        n = max(1, int(len(err) * (1.0 - reject)))
        rows.append({"reject_frac": reject, "n_kept": n, "mae": float(np.mean(err[:n]))})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lead = read_table(ROOT / "results" / "lead_skill.csv")
    preds = read_table(ROOT / "results" / "predictions.csv")
    panel = read_table(ROOT / "data_interim" / "obs_panel_labels.csv")
    trade_path = ROOT / "data_interim" / "obs_trade_network.csv"
    trade = read_table(trade_path) if trade_path.exists() else pd.DataFrame()

    stable = _stable_table(lead)
    write_table(stable, OUT / "stable_lead.csv")
    skill = _skill_vs_baselines(preds, panel)
    write_table(skill, OUT / "skill_scores.csv")
    sizes = _sample_sizes(preds, panel)
    (OUT / "sample_sizes.json").write_text(json.dumps(sizes, indent=2), encoding="utf-8")

    cov = _coverage_slices(preds)
    if len(cov):
        write_table(cov, OUT / "coverage_slices.csv")
    sel = _selective(preds)
    if len(sel):
        write_table(sel, OUT / "selective_prediction.csv")

    food = preds[preds["experiment"] == "P_food_system"].copy()
    trade_m = evaluate_trade(food, trade, 1, 0.80) if len(food) and len(trade) else pd.DataFrame()
    if len(trade_m):
        write_table(trade_m, OUT / "trade_metrics.csv")
        by_lead = trade_m.groupby("lead", as_index=False)[["recall", "precision", "f1", "auprc", "auroc"]].mean()
        harvest_ap = float(by_lead.loc[by_lead["lead"] == 0, "auprc"].mean()) if (by_lead["lead"] == 0).any() else np.nan
        by_lead["tews"] = [trade_early_warning_skill(float(r.auprc), harvest_ap) for r in by_lead.itertuples()]
        write_table(by_lead, OUT / "trade_metrics_by_lead.csv")

    outcome = (
        evaluate_realized_import_shortfall(food, trade, 1, 0.80, labels=panel)
        if len(food) and len(trade)
        else pd.DataFrame()
    )
    if len(outcome):
        write_table(outcome, OUT / "import_outcome.csv")
        out_lead = outcome.groupby("lead", as_index=False)[["recall", "precision", "f1", "auprc"]].mean()
        write_table(out_lead, OUT / "import_outcome_by_lead.csv")

    notes = [
        "# Revision P0 diagnostics (official wheat+soy caches)",
        "",
        "These numbers do **not** change official gates. Isolated `earliest_lead` stays the gate definition.",
        "",
        "## Claim boundary",
        "GOSIF is reconstructed from MODIS + meteorology. B4−B2=0 is an incremental-information null on GOSIF, not a proof that observed SIF has no lead. TROPOMI is still missing.",
        "",
        "## Stable lead vs isolated earliest_lead",
        stable.to_string(index=False),
        "",
        "## Sample sizes",
        json.dumps(sizes, indent=2),
        "",
        "## Skill scores (B2 vs zero-anomaly and persistence)",
        skill.to_string(index=False) if len(skill) else "n/a",
        "",
        "## Exposure-list metrics (model-implied high exposure)",
        (
            trade_m.groupby("lead")[["recall", "precision", "f1", "auprc"]].mean().sort_index(ascending=False).to_string()
            if len(trade_m)
            else "n/a"
        ),
        "",
        "## Realized importer quantity shortfall (actual-outcome test)",
        (
            outcome.groupby("lead")[["recall", "precision", "f1", "auprc"]].mean().sort_index(ascending=False).to_string()
            if len(outcome)
            else "n/a"
        ),
        "",
        "## Selective prediction (reject high OOD)",
        sel.to_string(index=False) if len(sel) else "n/a",
        "",
        "## Headline features",
        "Official wheat/soy series already come from crop-specific GEE points (`data_raw/gee_crop/`), not MCD12Q1 cropland means.",
        "",
        "## Not done here",
        "- TROPOMI/GOME-2 independent SIF",
        "- Retrain on Theil–Sen / first-difference / percent labels (`scripts/run_robustness.py` still defaults to quadratic)",
        "- Common-period 2015–2024 SMAP/VOD ablation",
        "- Mixture-of-experts for leave-one-region-out",
        "- Price, stocks, substitution, or export-restriction outcomes",
    ]
    (OUT / "README.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(stable.to_string(index=False))
    print(json.dumps(sizes, indent=2))
    if len(trade_m):
        print("exposure metrics")
        print(trade_m.groupby("lead")[["recall", "precision", "f1", "auprc"]].mean().sort_index(ascending=False))
    if len(outcome):
        print("import outcome")
        print(outcome.groupby("lead")[["recall", "precision", "f1", "auprc"]].mean().sort_index(ascending=False))
    if len(sel):
        print(sel.to_string(index=False))


if __name__ == "__main__":
    main()
