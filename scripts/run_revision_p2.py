"""Network baselines, importer-specific labels, persistence, optional 365 d official run.

Writes only to results/revision_p2.
"""

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
from cropwarn.phenology.alignment import cutoff_timestamp, harvest_doy
from cropwarn.trade.baselines import (
    evaluate_exposure_suite,
    lagged_optical_skill,
    permute_forecast_auprc,
    permutation_global,
    skill_fraction,
    year_block_auprc_ci,
    yield_persistence,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision_p2"
LONG_LEADS = [365, 320, 288, 256, 224, 192, 160, 128, 96, 64, 32, 0]


def feature_age_audit(series: pd.DataFrame, panel: pd.DataFrame, leads: list[int]) -> pd.DataFrame:
    rows = []
    meta = panel[["iso3", "crop", "year", "sos_true", "eos_true", "sos_static", "eos_static"]].drop_duplicates()
    s = series.copy()
    s["date"] = pd.to_datetime(s["date"])
    for _, m in meta.iterrows():
        sub = s[(s["iso3"] == m["iso3"]) & (s["crop"] == m["crop"]) & (s["year"] == m["year"])]
        if sub.empty:
            continue
        eos = m["eos_true"] if pd.notna(m.get("eos_true")) else m["eos_static"]
        sos = m["sos_true"] if pd.notna(m.get("sos_true")) else m["sos_static"]
        if not np.isfinite(eos) or not np.isfinite(sos):
            continue
        h = harvest_doy(int(eos))
        sow = pd.Timestamp(year=int(m["year"]), month=1, day=1) + pd.Timedelta(days=int(sos) - 1)
        if int(sos) > 200 and int(eos) < int(sos):
            sow = pd.Timestamp(year=int(m["year"]) - 1, month=1, day=1) + pd.Timedelta(days=int(sos) - 1)
        for lead in leads:
            cutoff_date = cutoff_timestamp(int(m["year"]), h, int(lead))
            pre = sub[sub["date"] <= cutoff_date]
            latest = pre["date"].max() if not pre.empty else pd.NaT
            age_vs_sow = (latest - sow).days if pd.notna(latest) else np.nan
            if not np.isfinite(age_vs_sow):
                season = "no_data"
            elif age_vs_sow < -60:
                season = "previous_season"
            elif age_vs_sow < 0:
                season = "preseason"
            else:
                season = "current_season"
            rows.append(
                {
                    "iso3": m["iso3"],
                    "crop": m["crop"],
                    "year": int(m["year"]),
                    "lead": int(lead),
                    "cutoff_date": cutoff_date,
                    "latest_optical": latest,
                    "days_after_sowing": age_vs_sow,
                    "before_sowing": bool(age_vs_sow < 0) if np.isfinite(age_vs_sow) else None,
                    "season_of_latest": season,
                }
            )
    return pd.DataFrame(rows)


def previous_year_evi_skill(series: pd.DataFrame, labels: pd.DataFrame) -> dict:
    return lagged_optical_skill(series, labels, var="evi")


def phase_diagnostics() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preds = read_table(ROOT / "results" / "predictions.csv")
    labels = read_table(ROOT / "data_interim" / "obs_panel_labels.csv")
    trade = read_table(ROOT / "data_interim" / "obs_trade_network.csv")
    series = read_table(ROOT / "data_interim" / "obs_season_series.csv")
    food = preds[preds["experiment"] == "P_food_system"].copy()
    if food.empty:
        food = preds[preds["experiment"] == "P_full_physiology"].copy()

    summaries = {}
    for mode in ("pooled", "importer_q", "importer_z"):
        suite = evaluate_exposure_suite(food, trade, labels, label_mode=mode)
        write_table(suite, OUT / f"exposure_suite_{mode}.csv")
        by = suite.groupby("lead", as_index=False)[
            [c for c in suite.columns if c.startswith("auprc_") or c.startswith("cont_") or c in ("n_high", "prevalence")]
        ].mean()
        write_table(by, OUT / f"exposure_suite_{mode}_by_lead.csv")
        row192 = by[by["lead"] == 192]
        def _v(frame, col):
            if frame.empty or col not in frame.columns:
                return float("nan")
            return float(frame[col].iloc[0])

        summaries[mode] = {
            "by_lead": by.to_dict("records"),
            "by_crop_192": suite[suite["lead"] == 192]
            .groupby("crop", as_index=False)[[c for c in suite.columns if c.startswith("auprc_")]]
            .mean()
            .to_dict("records"),
            "ci_192_forecast": year_block_auprc_ci(suite, "auprc_forecast", 192),
            "ci_0_forecast": year_block_auprc_ci(suite, "auprc_forecast", 0),
            "ci_192_n0": year_block_auprc_ci(suite, "auprc_n0_lag", 192),
            "ci_192_n1": year_block_auprc_ci(suite, "auprc_n1_mean", 192),
            "skill_fraction_vs_n0": skill_fraction(
                _v(row192, "auprc_forecast"), _v(row192, "auprc_n0_lag"), _v(row192, "auprc_oracle")
            ),
            "skill_fraction_vs_prevalence": skill_fraction(
                _v(row192, "auprc_forecast"), _v(row192, "prevalence"), _v(row192, "auprc_oracle")
            ),
        }

    for mode in ("pooled", "importer_q"):
        perm = permute_forecast_auprc(food, trade, labels, lead=192, n_perm=400, label_mode=mode)
        write_table(perm, OUT / f"permutation_192_{mode}.csv")
        summaries[f"permutation_{mode}"] = {
            **permutation_global(perm),
            "by_crop": perm.groupby("crop")[["auprc_real", "auprc_perm_mean", "p_ge_real"]].mean().to_dict("index")
            if len(perm)
            else {},
        }
    summaries["persistence_yield"] = yield_persistence(labels)
    summaries["previous_year_evi"] = previous_year_evi_skill(series, labels)

    series_age = series
    pre365 = ROOT / "results" / "revision_p1" / "obs_season_series_pre365.csv"
    if pre365.exists():
        series_age = read_table(pre365)
    age = feature_age_audit(
        series_age[series_age["crop"].isin(["wheat", "soybean"])],
        labels[labels["crop"].isin(["wheat", "soybean"])],
        [365, 256, 192, 96, 32, 0],
    )
    if len(age):
        write_table(age, OUT / "feature_age_rows.csv")
        season_share = (
            age.groupby(["lead", "season_of_latest"], as_index=False)
            .size()
            .pivot(index="lead", columns="season_of_latest", values="size")
            .fillna(0)
        )
        season_share = season_share.div(season_share.sum(axis=1), axis=0).reset_index()
        agg = age.groupby("lead", as_index=False).agg(
            frac_before_sowing=("before_sowing", "mean"),
            median_days_after_sowing=("days_after_sowing", "median"),
            n=("iso3", "size"),
        )
        agg = agg.merge(season_share, on="lead", how="left")
        write_table(agg, OUT / "feature_age_by_lead.csv")
        summaries["feature_age"] = agg.to_dict("records")

    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summaries, indent=2, default=str))


def phase_horizon() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_config(ROOT)
    raw = dict(base.raw)
    raw["study"] = dict(raw["study"])
    raw["study"]["headline_crops"] = ["wheat", "soybean"]
    raw["study"]["demo_leads_days"] = LONG_LEADS
    raw["paths"] = dict(raw["paths"])
    raw["paths"]["results"] = "results/revision_p2/horizon365"
    cfg = Config(raw=raw, root=base.root)
    trade = read_table(ROOT / "data_interim" / "obs_trade_network.csv")
    world = build_observational_world(ROOT, cfg.years, cfg.crops, trade=trade, preseason_days=365)
    bundle = run_experiment_suite(
        world,
        cfg,
        demo=True,
        schemes=["rolling_origin"],
        results_dir=ROOT / "results" / "revision_p2" / "horizon365",
        experiments=["B2_optical_climate", "B4_plus_sif", "P_food_system"],
    )
    write_table(bundle["lead_skill"], OUT / "horizon365_lead_skill.csv")
    print(bundle["lead_skill"][["experiment", "lead", "r2"]].to_string())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("diag", "horizon", "both"), default="diag")
    args = p.parse_args()
    if args.phase in ("diag", "both"):
        phase_diagnostics()
    if args.phase in ("horizon", "both"):
        phase_horizon()


if __name__ == "__main__":
    main()
