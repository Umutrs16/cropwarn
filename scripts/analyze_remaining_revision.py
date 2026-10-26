"""Season progress, ENSO baseline, cluster bootstrap, trade-structure outcomes."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.data.enso import enso_baseline_predictions, preseason_oni
from cropwarn.data.io import read_table, write_table
from cropwarn.phenology.alignment import harvest_doy, season_progress_signed, stage_name
from cropwarn.trade.network_outcomes import evaluate_structure_outcomes
from cropwarn.validation.bootstrap import cluster_bootstrap_lead
from cropwarn.validation.metrics import regression_metrics, skill_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision_p0"


def _attach_progress(preds: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("iso3", "crop", "year") if c in preds.columns]
    meta = panel[keys + [c for c in ("sos_true", "eos_true", "sos_static", "eos_static") if c in panel.columns]]
    meta = meta.drop_duplicates()
    p = preds.merge(meta, on=keys, how="left")
    sos = pd.to_numeric(p.get("sos_true"), errors="coerce").fillna(pd.to_numeric(p.get("sos_static"), errors="coerce"))
    eos = pd.to_numeric(p.get("eos_true"), errors="coerce").fillna(pd.to_numeric(p.get("eos_static"), errors="coerce"))
    lead = pd.to_numeric(p["lead"], errors="coerce")
    progress = []
    for s, e, L in zip(sos.to_numpy(), eos.to_numpy(), lead.to_numpy()):
        if not np.isfinite(s) or not np.isfinite(e) or not np.isfinite(L):
            progress.append(np.nan)
            continue
        cutoff = harvest_doy(int(e)) - int(L)
        if cutoff < 1:
            cutoff += 365
        progress.append(season_progress_signed(int(cutoff), int(s), int(e)))
    p["season_progress"] = progress
    p["stage"] = [stage_name(x) if np.isfinite(x) else "unknown" for x in progress]
    return p


def _stage_skill(preds: pd.DataFrame) -> pd.DataFrame:
    ycol = "yield_anomaly" if "yield_anomaly" in preds.columns else "y"
    rows = []
    for (exp, stage), g in preds.groupby(["experiment", "stage"]):
        rec = regression_metrics(g[ycol].to_numpy(), g["pred"].to_numpy())
        rec.update({"experiment": exp, "stage": stage, "n": int(rec["n"]) if np.isfinite(rec["n"]) else 0})
        rows.append(rec)
    return pd.DataFrame(rows)


def _write_stage_figure(skill: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    order = ["pre_sowing", "sowing_emergence", "vegetative", "flowering", "grain_filling", "maturity"]
    keep = ["B2_optical_climate", "B4_plus_sif", "B3_plus_sm", "B5_plus_vod"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(order))
    width = 0.2
    for i, exp in enumerate(keep):
        sub = skill[skill["experiment"] == exp]
        ys = [float(sub.loc[sub["stage"] == st, "r2"].mean()) if (sub["stage"] == st).any() else np.nan for st in order]
        ax.bar(x + (i - 1.5) * width, ys, width=width, label=exp.replace("_", " "))
    ax.axhline(0.15, color="0.4", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["pre-sow", "sow–emerg", "vegetative", "flowering", "grain fill", "maturity"], rotation=20)
    ax.set_ylabel("Pooled $R^2$")
    ax.set_title("Yield-anomaly skill by season progress (wheat + soybean)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preds = read_table(ROOT / "results" / "predictions.csv")
    panel = read_table(ROOT / "data_interim" / "obs_panel_labels.csv")
    trade = read_table(ROOT / "data_interim" / "obs_trade_network.csv")

    staged = _attach_progress(preds, panel)
    write_table(staged[["experiment", "iso3", "crop", "year", "lead", "season_progress", "stage"]], OUT / "season_progress_rows.csv")
    stage_skill = _stage_skill(staged)
    write_table(stage_skill, OUT / "season_progress_skill.csv")
    _write_stage_figure(stage_skill, ROOT / "figures" / "fig_season_progress.png")
    _write_stage_figure(stage_skill, OUT / "fig_season_progress.png")

    oni = preseason_oni(ROOT)
    enso = enso_baseline_predictions(panel[panel["crop"].isin(["wheat", "soybean"])], oni)
    write_table(enso, OUT / "enso_predictions.csv")
    enso_m = regression_metrics(enso["yield_anomaly"].to_numpy(), enso["pred"].to_numpy()) if len(enso) else {}
    b2 = preds[(preds["experiment"] == "B2_optical_climate") & (preds["lead"] == 192)]
    b2_m = regression_metrics(b2["yield_anomaly"].to_numpy(), b2["pred"].to_numpy()) if len(b2) else {}
    mse_enso = float(np.nanmean((enso["yield_anomaly"] - enso["pred"]) ** 2)) if len(enso) else np.nan
    mse_b2 = float(np.nanmean((b2["yield_anomaly"] - b2["pred"]) ** 2)) if len(b2) else np.nan
    mse0 = float(np.nanmean(b2["yield_anomaly"] ** 2)) if len(b2) else np.nan
    enso_summary = {
        "enso_r2": enso_m.get("r2"),
        "enso_n": enso_m.get("n"),
        "b2_192_r2": b2_m.get("r2"),
        "b2_192_n": b2_m.get("n"),
        "ss_enso_vs_zero": skill_score(mse_enso, float(np.nanmean(enso["yield_anomaly"] ** 2))) if len(enso) else None,
        "ss_b2_192_vs_zero": skill_score(mse_b2, mse0),
        "ss_b2_vs_enso": skill_score(mse_b2, mse_enso),
    }
    (OUT / "enso_vs_b2.json").write_text(json.dumps(enso_summary, indent=2), encoding="utf-8")

    boot = cluster_bootstrap_lead(preds, n_boot=400, seed=42)
    write_table(boot, OUT / "cluster_bootstrap.csv")

    food = preds[preds["experiment"] == "P_food_system"]
    struct = evaluate_structure_outcomes(food, trade)
    if len(struct):
        write_table(struct, OUT / "trade_structure.csv")
        by = struct.groupby(["outcome", "lead"], as_index=False)[["recall", "auprc"]].mean()
        write_table(by, OUT / "trade_structure_by_lead.csv")

    notes = {
        "season_progress": "See season_progress_skill.csv and figures/fig_season_progress.png",
        "enso": enso_summary,
        "bootstrap_experiments": boot["experiment"].tolist() if len(boot) else [],
        "trade_structure_outcomes": sorted(struct["outcome"].unique().tolist()) if len(struct) else [],
    }
    (OUT / "remaining_summary.json").write_text(json.dumps(notes, indent=2, default=str), encoding="utf-8")
    print(stage_skill.pivot_table(index="stage", columns="experiment", values="r2").to_string())
    print(json.dumps(enso_summary, indent=2))
    print(boot.to_string(index=False) if len(boot) else "no bootstrap")
    if len(struct):
        print(struct.groupby("outcome")[["recall", "auprc"]].mean())


if __name__ == "__main__":
    main()
