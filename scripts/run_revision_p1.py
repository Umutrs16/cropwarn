"""P0 revision suite. Writes only to results/revision_p1 — never official results/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.config import Config, load_config
from cropwarn.data.io import read_table, write_table
from cropwarn.data.observational import build_observational_world
from cropwarn.experiments.runner import run_experiment_suite
from cropwarn.trade.exposure import evaluate_expost_exposure
from cropwarn.validation.bootstrap import cluster_bootstrap_lead
from cropwarn.validation.metrics import earliest_reliable_lead, regression_metrics, robust_stable_lead, stable_reliable_lead

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision_p1"
C_EXPS = ["C0_climatology", "C1_climate", "C2_optical", "C3_climate_optical", "C4_climate_optical_sif"]
LONG_LEADS = [365, 320, 288, 256, 224, 192, 160, 128, 96, 64, 32, 0]


def _lead_skill_by_crop(preds: pd.DataFrame) -> pd.DataFrame:
    ycol = "yield_anomaly" if "yield_anomaly" in preds.columns else "y"
    rows = []
    for (exp, crop, lead), g in preds.groupby(["experiment", "crop", "lead"]):
        rec = regression_metrics(g[ycol].to_numpy(), g["pred"].to_numpy())
        rec.update({"experiment": exp, "crop": crop, "lead": int(lead)})
        rows.append(rec)
    skill = pd.DataFrame(rows)
    erl = []
    for (exp, crop), g in skill.groupby(["experiment", "crop"]):
        erl.append(
            {
                "experiment": exp,
                "crop": crop,
                "isolated_lead": earliest_reliable_lead(g),
                "stable_lead": stable_reliable_lead(g),
            }
        )
    return skill.merge(pd.DataFrame(erl), on=["experiment", "crop"], how="left")


def _bounds_from_bootstrap(boot: pd.DataFrame, experiment: str) -> pd.DataFrame:
    row = boot[boot["experiment"] == experiment]
    if row.empty:
        return pd.DataFrame()
    rec = row.iloc[0]
    leads = []
    for col in rec.index:
        if col.startswith("r2_") and col.endswith("_p05"):
            lead = int(col.split("_")[1])
            leads.append({"lead": lead, "r2_p05": float(rec[col])})
    return pd.DataFrame(leads).sort_values("lead")


def phase_a() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preds = read_table(ROOT / "results" / "predictions.csv")
    labels = read_table(ROOT / "data_interim" / "obs_panel_labels.csv")
    trade = read_table(ROOT / "data_interim" / "obs_trade_network.csv")
    food = preds[preds["experiment"] == "P_food_system"].copy()
    if food.empty:
        food = preds[preds["experiment"] == "P_full_physiology"].copy()

    expost = evaluate_expost_exposure(food, trade, labels)
    write_table(expost, OUT / "expost_exposure.csv")
    by_lead = (
        expost.groupby("lead", as_index=False)[["recall", "precision", "auprc", "n_high", "cont_spearman", "cont_r2"]]
        .mean()
        .sort_values("lead")
        if len(expost)
        else pd.DataFrame()
    )
    write_table(by_lead, OUT / "expost_exposure_by_lead.csv")

    crop_skill = _lead_skill_by_crop(preds[preds["experiment"].isin(["B2_optical_climate", "B4_plus_sif", "P_food_system"])])
    write_table(crop_skill, OUT / "lead_skill_by_crop.csv")

    boot = cluster_bootstrap_lead(preds, n_boot=400, seed=42)
    write_table(boot, OUT / "cluster_bootstrap.csv")
    robust_rows = []
    for exp in ("B2_optical_climate", "B4_plus_sif"):
        bounds = _bounds_from_bootstrap(boot, exp)
        robust_rows.append(
            {
                "experiment": exp,
                "robust_stable_lead": robust_stable_lead(bounds) if len(bounds) else np.nan,
            }
        )
    write_table(pd.DataFrame(robust_rows), OUT / "robust_stable_lead.csv")

    summary = {
        "expost_192_auprc": float(by_lead.loc[by_lead["lead"] == 192, "auprc"].mean()) if len(by_lead) and 192 in set(by_lead["lead"]) else None,
        "expost_0_auprc": float(by_lead.loc[by_lead["lead"] == 0, "auprc"].mean()) if len(by_lead) and 0 in set(by_lead["lead"]) else None,
        "expost_192_spearman": float(by_lead.loc[by_lead["lead"] == 192, "cont_spearman"].mean()) if len(by_lead) and 192 in set(by_lead["lead"]) else None,
        "expost_192_n_high": float(by_lead.loc[by_lead["lead"] == 192, "n_high"].mean()) if len(by_lead) and 192 in set(by_lead["lead"]) else None,
        "robust_stable": robust_rows,
        "dstable": boot[boot["experiment"] == "B4_minus_B2"][
            [c for c in boot.columns if c.startswith("dstable") or c.startswith("disolated")]
        ].to_dict("records")
        if "B4_minus_B2" in set(boot["experiment"])
        else [],
        "by_crop": crop_skill[crop_skill["lead"].isin([0, 192])][
            ["experiment", "crop", "lead", "r2", "isolated_lead", "stable_lead"]
        ].to_dict("records"),
    }
    (OUT / "phase_a_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


def phase_b() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_config(ROOT)
    raw = dict(base.raw)
    raw["study"] = dict(raw["study"])
    raw["study"]["headline_crops"] = ["wheat", "soybean"]
    raw["study"]["demo_leads_days"] = LONG_LEADS
    raw["paths"] = dict(raw["paths"])
    raw["paths"]["results"] = "results/revision_p1/c_ablation"
    cfg = Config(raw=raw, root=base.root)
    trade = read_table(ROOT / "data_interim" / "obs_trade_network.csv")
    world = build_observational_world(ROOT, cfg.years, cfg.crops, trade=trade, preseason_days=365)
    write_table(world["series"], OUT / "obs_season_series_pre365.csv")
    print("preseason series", len(world["series"]), flush=True)
    bundle = run_experiment_suite(
        world,
        cfg,
        demo=True,
        schemes=["rolling_origin"],
        results_dir=ROOT / "results" / "revision_p1" / "c_ablation",
        experiments=C_EXPS,
    )
    lead = bundle["lead_skill"]
    write_table(lead, OUT / "c_lead_skill.csv")
    print(lead[["experiment", "lead", "r2"]].to_string())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("a", "b", "both"), default="a")
    args = p.parse_args()
    if args.phase in ("a", "both"):
        phase_a()
    if args.phase in ("b", "both"):
        phase_b()


if __name__ == "__main__":
    main()
