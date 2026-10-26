"""Ablation, shock-holdout and food-system experiment suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cropwarn.config import Config
from cropwarn.data.io import write_table
from cropwarn.features.factory import MODALITY_VARS, build_feature_table, feature_columns
from cropwarn.models.area import attach_area_states, production_interval
from cropwarn.models.predictors import fit_predict
from cropwarn.models.uncertainty import (
    OODDetector,
    assign_trust,
    crps_gaussian_proxy,
    empirical_coverage,
    split_conformal,
)
from cropwarn.phenology.alignment import PhenologyVersion
from cropwarn.trade.exposure import evaluate_trade
from cropwarn.validation.go_nogo import evaluate_gates, shock_year_lead_gain, write_report
from cropwarn.validation.leakage import assert_no_label_columns, audit_split
from cropwarn.validation.metrics import (
    classification_metrics,
    earliest_reliable_lead,
    regression_metrics,
    selective_risk,
)
from cropwarn.validation.splits import iter_splits


DROP_PREFIX = {
    "optical": ("ndvi_", "evi_", "lai_"),
    "thermal": ("lst_", "vpd_"),
    "climate": ("lst_", "vpd_", "precip_", "et0_"),
    "sm": ("rzsm_",),
    "sif": ("sif_",),
    "vod": ("vod_",),
}


def _mask_modalities(df: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    out = df.copy()
    drop_mods = [m for m in MODALITY_VARS if m not in keep]
    kill: list[str] = []
    for m in drop_mods:
        for pref in DROP_PREFIX[m]:
            kill.extend([c for c in out.columns if c.startswith(pref)])
    if kill:
        out = out.drop(columns=list(dict.fromkeys(kill)), errors="ignore")
    return out


def _subset_metrics(test: pd.DataFrame, y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    extra = {}
    for name, mask in {
        "lowlat_multicrop": (test["latitude_band"].isin(["tropical", "subtropical"])) & (test["multi_crop"]),
        "temperate": test["latitude_band"] == "temperate",
        "global_south": test["region"].isin(["Africa", "SouthAsia", "SoutheastAsia", "WestAsia"]),
    }.items():
        m = mask.to_numpy()
        extra[f"r2_{name}"] = regression_metrics(y[m], pred[m])["r2"]
    return extra


def _run_split(
    feat: pd.DataFrame,
    split,
    model_name: str,
    target: str,
    demo: bool,
    cfg: Config,
    conformal: bool,
) -> tuple[pd.DataFrame, dict]:
    train = feat.loc[split.train_idx]
    cal = feat.loc[split.cal_idx] if len(split.cal_idx) else train.iloc[0:0]
    test = feat.loc[split.test_idx]
    audit_split(train, cal if len(cal) else train.tail(0), test)
    assert_no_label_columns(feature_columns(train))

    pred = fit_predict(model_name, train, test, target, demo, cfg)
    y = test[target].to_numpy(dtype=float)
    y_lo, y_hi = np.nanpercentile(train[target], [1, 99])
    pred = np.clip(pred, y_lo - 3 * np.nanstd(train[target]), y_hi + 3 * np.nanstd(train[target]))
    if conformal and len(cal) >= 20:
        pred_cal = fit_predict(model_name, train, cal, target, demo, cfg)
        conf = split_conformal(cal[target].to_numpy(float), pred_cal, pred, float(cfg.get("validation", "conformal_alpha")))
    else:
        resid_scale = float(np.nanstd(train[target])) * 0.8
        conf = split_conformal(train[target].to_numpy(float)[:50], np.zeros(min(50, len(train))), pred, 0.10)
        conf.width[:] = resid_scale
        conf.lower = pred - resid_scale
        conf.upper = pred + resid_scale

    ood = OODDetector()
    ood.fit(train)
    ood_df = ood.scores(test)
    trust = assign_trust(ood_df["ood_score"].to_numpy(), float(cfg.get("validation", "ood_reject_quantile")))
    abs_err = np.abs(y - pred)
    cat = float(np.mean(abs_err > float(cfg.get("validation", "catastrophic_rel_error")) * (np.abs(y) + 0.25)))

    rec = {
        "split": split.name,
        "scheme": split.scheme,
        "test_year": split.test_year,
        "heldout": split.heldout,
        **regression_metrics(y, pred),
        **classification_metrics(y, pred),
        "coverage90": empirical_coverage(y, conf.lower, conf.upper),
        "mean_width": float(np.nanmean(conf.width)),
        "crps": crps_gaussian_proxy(y, pred, conf.width),
        "catastrophic_rate": cat,
        **selective_risk(abs_err, ood_df["ood_score"].to_numpy()),
        **_subset_metrics(test, y, pred),
    }

    keep_cols = [c for c in ("iso3", "unit_id", "crop", "year", "lead", "region", "latitude_band", "multi_crop", "exporter", "harvested_ha", "shock_true", target) if c in test.columns]
    pred_tbl = test[keep_cols].copy()
    pred_tbl["pred"] = pred
    pred_tbl["lower"] = conf.lower
    pred_tbl["upper"] = conf.upper
    pred_tbl["ood_score"] = ood_df["ood_score"].to_numpy()
    pred_tbl["trust"] = trust
    pred_tbl["abs_err"] = abs_err
    pred_tbl["split"] = split.name
    pred_tbl["scheme"] = split.scheme
    return pred_tbl, rec


def run_experiment_suite(
    world: dict[str, pd.DataFrame],
    cfg: Config,
    demo: bool = True,
    schemes: list[str] | None = None,
    results_dir: Path | None = None,
    experiments: list[str] | None = None,
) -> dict[str, pd.DataFrame | dict]:
    panel = attach_area_states(world["panel"])
    series = world["series"]
    headline = cfg.get("study", "headline_crops")
    if headline:
        panel = panel[panel["crop"].isin(headline)].copy()
        if "crop" in series.columns:
            series = series[series["crop"].isin(headline)].copy()
    trade = world["trade"]
    leads = cfg.leads(demo=demo)
    target = str(cfg.get("study", "primary_target"))
    schemes = schemes or (["rolling_origin"] if demo else None)
    # Demo still uses the short lead suite, but every rolling-origin year from
    # rolling_test_start. A 3-year slice overweighting 2020 is not a gate.
    demo_years = None

    caches: dict[str, pd.DataFrame] = {}
    for version in (PhenologyVersion.STATIC, PhenologyVersion.DYNAMIC):
        caches[version.value] = build_feature_table(
            panel, series, leads, version, list(MODALITY_VARS)
        )

    ablations = cfg.get("ablations")
    if experiments:
        ablations = {k: v for k, v in ablations.items() if k in experiments}
    metric_rows = []
    pred_rows = []
    for exp_name, spec in ablations.items():
        feat = _mask_modalities(caches[spec["phenology"]], spec["modalities"])
        # Evaluate each lead separately so early-warning skill curves are honest.
        for lead, block in feat.groupby("lead"):
            block = block.reset_index(drop=True)
            block = block[np.isfinite(block[target])].reset_index(drop=True)
            for split in iter_splits(block, cfg, schemes=schemes):
                if demo_years is not None and split.test_year not in demo_years:
                    continue
                preds, rec = _run_split(
                    block,
                    split,
                    spec["model"],
                    target,
                    demo,
                    cfg,
                    bool(spec["conformal"]),
                )
                rec.update({"experiment": exp_name, "lead": int(lead), "model": spec["model"], "phenology": spec["phenology"]})
                metric_rows.append(rec)
                preds["experiment"] = exp_name
                pred_rows.append(preds)

    metrics = pd.DataFrame(metric_rows)
    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()

    agg = {
        "r2": ("r2", "mean"),
        "rmse": ("rmse", "mean"),
        "auroc": ("auroc", "mean"),
        "coverage90": ("coverage90", "mean"),
        "mean_width": ("mean_width", "mean"),
    }
    if "r2_lowlat_multicrop" in metrics.columns:
        agg["r2_lowlat_multicrop"] = ("r2_lowlat_multicrop", "mean")
    gate_src = metrics
    if "scheme" in metrics.columns and (metrics["scheme"] == "rolling_origin").any():
        gate_src = metrics[metrics["scheme"] == "rolling_origin"]
    lead_skill = gate_src.groupby(["experiment", "lead", "phenology"], as_index=False).agg(**agg)
    erl = []
    for exp, g in lead_skill.groupby("experiment"):
        erl.append({"experiment": exp, "earliest_lead": earliest_reliable_lead(g)})
    lead_skill = lead_skill.merge(pd.DataFrame(erl), on="experiment", how="left")

    trade_tbl = pd.DataFrame()
    if not preds.empty:
        food = preds[preds["experiment"] == "P_food_system"].copy()
        if food.empty:
            food = preds[preds["experiment"] == "P_full_physiology"].copy()
        if not food.empty:
            trade_tbl = evaluate_trade(
                food,
                trade,
                int(cfg.get("trade", "network_lag_years")),
                float(cfg.get("trade", "high_exposure_quantile")),
            )

    trade_gain = np.nan
    food_skill = lead_skill[lead_skill["experiment"].isin(["P_food_system", "P_full_physiology"])]
    harvest_r2 = (
        float(food_skill.loc[food_skill["lead"] == 0, "r2"].mean()) if len(food_skill) else np.nan
    )
    harvest_lead = earliest_reliable_lead(food_skill) if len(food_skill) else np.nan
    if not trade_tbl.empty and np.isfinite(harvest_lead) and np.isfinite(harvest_r2) and harvest_r2 >= 0.15:
        by_lead = trade_tbl.groupby("lead")["recall"].mean()
        if len(by_lead) >= 2:
            # Days of lead at which recall first reaches 80% of harvest-time recall.
            harvest = float(by_lead.min() if 0 not in by_lead.index else by_lead[0])
            target_r = 0.8 * harvest if harvest == harvest else 0.5
            ok = by_lead[by_lead >= target_r]
            trade_gain = float(ok.index.max()) if len(ok) else 0.0

    # Production intervals for the full model
    prod = pd.DataFrame()
    full = preds[preds["experiment"] == "P_full_physiology"]
    if not full.empty:
        yhat = full["pred"].to_numpy()
        # Convert anomaly predictions back to a production proxy via harvested area.
        width = (full["upper"] - full["lower"]).to_numpy()
        prod = production_interval(
            np.abs(yhat) + 1.0,
            width,
            full["harvested_ha"].to_numpy(),
            draws=int(cfg.get("area", "mc_draws")) if not demo else 120,
            seed=cfg.seed,
        )
        prod = pd.concat([full[["iso3", "crop", "year", "lead"]].reset_index(drop=True), prod], axis=1)

    gates = evaluate_gates(lead_skill, cfg, trade_gain)
    gates["shock_year_physiology"] = shock_year_lead_gain(preds, panel, cfg)

    results_dir = Path(results_dir) if results_dir is not None else cfg.path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    write_table(metrics, results_dir / "metrics.csv")
    write_table(lead_skill, results_dir / "lead_skill.csv")
    write_table(preds, results_dir / "predictions.csv")
    if not trade_tbl.empty:
        write_table(trade_tbl, results_dir / "trade_exposure.csv")
    if not prod.empty:
        write_table(prod, results_dir / "production_intervals.csv")
    write_report(gates, results_dir / "go_nogo.json")

    return {
        "metrics": metrics,
        "lead_skill": lead_skill,
        "predictions": preds,
        "trade": trade_tbl,
        "production": prod,
        "features_static": caches["static"],
        "features_dynamic": caches["dynamic"],
        "panel": panel,
        "series": series,
        "gates": gates,
    }
