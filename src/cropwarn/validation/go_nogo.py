"""Internal Nature-tier Go/No-Go gates from section 6.4."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate_gates(summary: pd.DataFrame, cfg, trade_gain_days: float | None = None) -> dict:
    gates = cfg.get("go_nogo")
    experiments = set(summary["experiment"]) if "experiment" in summary.columns else set()
    phenology_gain = float("nan")
    if {"B0_chen_rf", "B1_dynamic_rf"} <= experiments:
        pheno_col = "r2_lowlat_multicrop" if "r2_lowlat_multicrop" in summary.columns else "r2"
        b0 = summary.loc[summary["experiment"] == "B0_chen_rf", pheno_col]
        b1 = summary.loc[summary["experiment"] == "B1_dynamic_rf", pheno_col]
        b0, b1 = b0.dropna(), b1.dropna()
        if len(b0) and len(b1):
            phenology_gain = float(b1.mean() - b0.mean())
        if not np.isfinite(phenology_gain):
            b0 = summary.loc[summary["experiment"] == "B0_chen_rf", "r2"].dropna()
            b1 = summary.loc[summary["experiment"] == "B1_dynamic_rf", "r2"].dropna()
            if len(b0) and len(b1):
                phenology_gain = float(b1.mean() - b0.mean())

    # Same-model ablation isolates the physiology increment (B4 XGB vs B2 XGB).
    # earliest_lead is days before harvest with R² >= 0.15; do not invent a lead
    # from a small AUROC difference when that floor is never met.
    phys = summary[summary["experiment"] == "B4_plus_sif"]
    opt = summary[summary["experiment"] == "B2_optical_climate"]
    if phys.empty:
        phys = summary[summary["experiment"] == "P_full_physiology"]
    if opt.empty:
        opt = summary[summary["experiment"] == "B0_chen_rf"]

    def _mean_lead(frame: pd.DataFrame) -> float:
        if frame.empty or "earliest_lead" not in frame.columns:
            return float("nan")
        s = pd.to_numeric(frame["earliest_lead"], errors="coerce").dropna()
        s = s[np.isfinite(s)]
        return float(s.mean()) if len(s) else float("nan")

    lead_gain = float("nan")
    p_lead, o_lead = _mean_lead(phys), _mean_lead(opt)
    if np.isfinite(p_lead) and np.isfinite(o_lead):
        lead_gain = float(p_lead - o_lead)

    coverage_err = float("nan")
    if "coverage90" in summary:
        cov = summary
        if "experiment" in summary.columns:
            conf = summary[summary["experiment"].isin(["P_full_physiology", "P_food_system"])]
            if len(conf):
                cov = conf
        coverage_err = float((cov["coverage90"] - 0.90).abs().mean())

    pheno_abs = float("nan")
    if {"B0_chen_rf", "B1_dynamic_rf"} <= experiments:
        pheno_col_abs = "r2_lowlat_multicrop" if "r2_lowlat_multicrop" in summary.columns else "r2"
        b1_abs = summary.loc[summary["experiment"] == "B1_dynamic_rf", pheno_col_abs].dropna()
        if len(b1_abs):
            pheno_abs = float(b1_abs.mean())

    report = {
        "dynamic_phenology": {
            "value": phenology_gain,
            "threshold": gates["phenology_lowlat_r2_gain"],
            "go": bool(
                np.isfinite(phenology_gain)
                and phenology_gain >= gates["phenology_lowlat_r2_gain"]
                and np.isfinite(pheno_abs)
                and pheno_abs > 0.0
            ),
            "fallback": "Treat as a phenology-bias paper or drop as the headline innovation.",
        },
        "physiology_lead": {
            "value": lead_gain,
            "threshold": gates["physiology_lead_days"],
            "go": bool(lead_gain == lead_gain and lead_gain >= gates["physiology_lead_days"]),
            "fallback": "If only harvest-time R2 rises, position as RSE not Nature Food.",
        },
        "uncertainty": {
            "value": coverage_err,
            "threshold": gates["coverage_tolerance"],
            "go": bool(coverage_err == coverage_err and coverage_err <= gates["coverage_tolerance"]),
            "fallback": "Do not put trustworthy in the title if intervals are uncalibrated.",
        },
        "trade_value": {
            "value": trade_gain_days,
            "threshold": gates["trade_lead_gain_days"],
            "go": bool(
                trade_gain_days is not None
                and trade_gain_days == trade_gain_days
                and trade_gain_days >= gates["trade_lead_gain_days"]
            ),
            "fallback": "If trade is only a schematic, move the story to Nat Commun / RSE.",
        },
    }
    report["overall_go"] = all(v["go"] for k, v in report.items() if k != "overall_go")
    return report


def shock_year_lead_gain(preds: pd.DataFrame, panel: pd.DataFrame, cfg) -> dict:
    """B4 vs B2 lead on documented event years / |z|>=1 rows. Not an official gate."""

    from cropwarn.validation.metrics import earliest_reliable_lead, regression_metrics

    empty = {"value": float("nan"), "b4_lead": float("nan"), "b2_lead": float("nan"), "n": 0, "go": False}
    if preds is None or preds.empty:
        return empty
    years = {int(s["year"]) for s in (cfg.get("shocks") or [])}
    keys = [c for c in ("iso3", "unit_id", "crop", "year") if c in preds.columns and c in panel.columns]
    p = preds.copy()
    if "yield_z" in panel.columns and keys:
        z = panel[keys + ["yield_z"]].drop_duplicates()
        p = p.merge(z, on=keys, how="left")
    mask = p["year"].isin(years) if years else pd.Series(False, index=p.index)
    if "yield_z" in p.columns:
        mask = mask | (pd.to_numeric(p["yield_z"], errors="coerce").abs() >= 1.0)
    p = p[mask]
    if p.empty or "pred" not in p.columns:
        return empty
    target = "y" if "y" in p.columns else ("yield_anomaly" if "yield_anomaly" in p.columns else None)
    if target is None:
        return empty
    rows = []
    for (exp, lead), g in p.groupby(["experiment", "lead"]):
        rec = regression_metrics(g[target].to_numpy(), g["pred"].to_numpy())
        rec.update({"experiment": exp, "lead": int(lead), "r2": rec["r2"]})
        rows.append(rec)
    skill = pd.DataFrame(rows)
    if skill.empty:
        return empty

    def _lead(name: str) -> float:
        sub = skill[skill["experiment"] == name]
        return earliest_reliable_lead(sub) if len(sub) else float("nan")

    b4 = _lead("B4_plus_sif")
    b2 = _lead("B2_optical_climate")
    gain = float(b4 - b2) if np.isfinite(b4) and np.isfinite(b2) else float("nan")
    thresh = float(cfg.get("go_nogo", "physiology_lead_days") or 14)
    keys_cy = [c for c in ("iso3", "unit_id", "crop", "year") if c in p.columns]
    n_cy = int(p[keys_cy].drop_duplicates().shape[0]) if keys_cy else int(p[["year"]].drop_duplicates().shape[0])
    n_cty = int(p["iso3"].nunique()) if "iso3" in p.columns else 0
    return {
        "value": gain,
        "b4_lead": b4,
        "b2_lead": b2,
        "n": int(len(p)),
        "n_prediction_rows": int(len(p)),
        "n_country_crop_years": n_cy,
        "n_countries": n_cty,
        "threshold": thresh,
        "go": bool(np.isfinite(gain) and gain >= thresh),
    }


def supported_journal(report: dict) -> str:
    """Journal position implied by official gates. Does not change gate thresholds."""

    phys = report.get("physiology_lead", {})
    trade = report.get("trade_value", {})
    pheno = report.get("dynamic_phenology", {})
    phys_go = bool(isinstance(phys, dict) and phys.get("go"))
    trade_go = bool(isinstance(trade, dict) and trade.get("go"))
    pheno_go = bool(isinstance(pheno, dict) and pheno.get("go"))
    if phys_go and trade_go:
        return "Nature Food"
    if trade_go and not phys_go:
        return "Nature Communications / RSE"
    if pheno_go and not phys_go:
        return "Remote Sensing of Environment / CEE"
    return "Remote Sensing of Environment"


def write_report(report: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = ["# Go / No-Go report", ""]
    for key, val in report.items():
        if key == "overall_go":
            md.append(f"**Nature Food overall:** {'GO' if val else 'NO-GO'}")
            md.append(f"**Supported journal:** {supported_journal(report)}")
            continue
        if not isinstance(val, dict) or "go" not in val:
            continue
        flag = "GO" if val["go"] else "NO-GO"
        md.append(f"- **{key}** [{flag}] value={val.get('value')} threshold={val.get('threshold')}")
        if not val["go"] and val.get("fallback"):
            md.append(f"  - {val['fallback']}")
    path.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    return path
