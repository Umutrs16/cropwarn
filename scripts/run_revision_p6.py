"""P0 statistical and temporal robustness. Writes only to results/revision_p6/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.config import load_config
from cropwarn.data.io import read_table, write_table
from cropwarn.data.labels import rolling_trend_anomaly
from cropwarn.features.factory import build_feature_table
from cropwarn.models.predictors import fit_predict
from cropwarn.phenology.alignment import PhenologyVersion
from cropwarn.trade.baselines import (
    _auprc,
    evaluate_exposure_suite,
    exporter_information_ablation,
    pooled_shock_permutation,
    year_block_auprc_ci,
)
from cropwarn.trade.buffers import skill_score
from cropwarn.trade.exposure import (
    import_rolling_anomaly,
    importer_exposure_rows,
    realized_import_change,
)
from cropwarn.validation.metrics import regression_metrics
from cropwarn.validation.splits import rolling_origin

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision_p6"
CROPS = ("wheat", "soybean")
N_BOOT = 2000
N_PERM = 2000


def production_obs_shocks(labels: pd.DataFrame) -> pd.DataFrame:
    lab = labels.copy()
    if "production_t" not in lab.columns:
        lab["production_t"] = pd.to_numeric(lab["yield_tha"], errors="coerce") * pd.to_numeric(
            lab["harvested_ha"], errors="coerce"
        )
    anom = rolling_trend_anomaly(lab, value_col="production_t", method="percent")
    out = anom[["iso3", "crop", "year"]].copy()
    out["shock_obs"] = pd.to_numeric(anom["yield_anomaly"], errors="coerce")
    return out


def _ci_row(suite: pd.DataFrame, col: str, lead: int) -> dict[str, float]:
    return year_block_auprc_ci(suite, col, lead, n_boot=N_BOOT)


def summarize_suite(suite: pd.DataFrame, name: str) -> dict:
    if suite.empty:
        return {"name": name}
    by = suite.groupby("lead", as_index=False)[
        [c for c in suite.columns if c.startswith("auprc_") or c.startswith("cont_") or c in ("prevalence", "n_high")]
    ].mean()
    write_table(by, OUT / f"{name}_by_lead.csv")
    s192 = suite[suite["lead"] == 192]
    crop192 = (
        s192.groupby("crop", as_index=False)[[c for c in suite.columns if c.startswith("auprc_") or c.startswith("cont_")]]
        .mean()
        if len(s192)
        else pd.DataFrame()
    )
    if len(crop192):
        write_table(crop192, OUT / f"{name}_by_crop_192.csv")
    row192 = by[by["lead"] == 192]
    def _v(col):
        if row192.empty or col not in row192.columns:
            return float("nan")
        return float(row192[col].iloc[0])

    return {
        "name": name,
        "auprc_192": _v("auprc_forecast"),
        "spearman_192": _v("cont_spearman"),
        "prevalence_192": _v("prevalence"),
        "n0_192": _v("auprc_n0_lag"),
        "n1_192": _v("auprc_n1_mean"),
        "n2_192": _v("auprc_n2_hhi"),
        "n3_192": _v("auprc_n3_propensity"),
        "oracle_192": _v("auprc_oracle"),
        "ci_forecast_192": _ci_row(suite, "auprc_forecast", 192),
        "ci_spearman_192": _ci_row(suite, "cont_spearman", 192),
        "ci_n0_192": _ci_row(suite, "auprc_n0_lag", 192),
        "ci_n3_192": _ci_row(suite, "auprc_n3_propensity", 192),
        "by_crop_192": crop192.to_dict("records") if len(crop192) else [],
    }


def run_endpoint_suite(
    food,
    trade,
    labels,
    obs_shocks,
    tag: str,
    lag_years: int = 1,
    do_perm: bool = True,
    do_ablate: bool = True,
) -> tuple[pd.DataFrame, dict]:
    suite = evaluate_exposure_suite(
        food, trade, labels, lag_years=lag_years, label_mode="pooled", obs_shocks=obs_shocks
    )
    write_table(suite, OUT / f"suite_{tag}.csv")
    summary = summarize_suite(suite, f"suite_{tag}")
    if do_perm:
        perm_draws, perm = pooled_shock_permutation(
            food,
            trade,
            labels,
            lead=192,
            n_perm=N_PERM,
            lag_years=lag_years,
            label_mode="pooled",
            obs_shocks=obs_shocks,
        )
        write_table(perm_draws, OUT / f"perm_{tag}.csv")
        summary["permutation"] = perm
    if do_ablate:
        ablate = exporter_information_ablation(
            food, trade, labels, lead=192, lag_years=lag_years, obs_shocks=obs_shocks, drop_from_label=False
        )
        write_table(ablate, OUT / f"ablation_{tag}.csv")
        summary["ablation_range"] = [
            float(ablate.loc[ablate["dropped"] != "none", "auprc"].min()) if len(ablate) > 1 else np.nan,
            float(ablate.loc[ablate["dropped"] != "none", "auprc"].max()) if len(ablate) > 1 else np.nan,
        ]
    return suite, summary


def shortfall_sensitivity(food, trade, labels) -> pd.DataFrame:
    expo = importer_exposure_rows(food[food["lead"] == 192], trade, labels, lag_years=1)
    if expo.empty:
        return pd.DataFrame()
    yoy = realized_import_change(trade)
    anom = import_rolling_anomaly(trade)
    expo = expo.merge(yoy[["year", "crop", "importer", "import_chg"]], on=["year", "crop", "importer"], how="left")
    expo = expo.merge(anom[["year", "crop", "importer", "import_anomaly"]], on=["year", "crop", "importer"], how="left")
    rows = []
    for (year, crop), g in expo.groupby(["year", "crop"]):
        score = pd.to_numeric(g["exposure_hat"], errors="coerce")
        for name, col in (("yoy", "import_chg"), ("anomaly", "import_anomaly")):
            y = pd.to_numeric(g[col], errors="coerce")
            m = score.notna() & y.notna()
            if int(m.sum()) < 8:
                continue
            rho = float("nan")
            try:
                rho = float(spearmanr(score[m], y[m]).statistic)
            except Exception:
                rho = float("nan")
            rec = {
                "year": int(year),
                "crop": crop,
                "definition": name,
                "spearman": rho,
                "n": float(m.sum()),
            }
            for q in (0.10, 0.20, 0.30):
                truth = y <= y.quantile(q)
                rec[f"auprc_q{int(100 * q)}"] = _auprc(truth.reset_index(drop=True), score.reset_index(drop=True))
                rec[f"pi_q{int(100 * q)}"] = float(truth.mean())
            rows.append(rec)
    out = pd.DataFrame(rows)
    write_table(out, OUT / "shortfall_sensitivity_year.csv")
    by = out.groupby("definition", as_index=False)[
        [c for c in out.columns if c.startswith("auprc_") or c.startswith("pi_") or c == "spearman"]
    ].mean()
    write_table(by, OUT / "shortfall_sensitivity.csv")
    return by


def expansion_exporters(labels: pd.DataFrame) -> pd.DataFrame:
    lab = labels[labels["crop"].isin(CROPS)].copy()
    lab["harvested_ha"] = pd.to_numeric(lab["harvested_ha"], errors="coerce")
    early = lab[lab["year"].between(2001, 2005)].groupby(["iso3", "crop"])["harvested_ha"].mean()
    late = lab[lab["year"].between(2016, 2020)].groupby(["iso3", "crop"])["harvested_ha"].mean()
    tab = pd.concat([early.rename("area_early"), late.rename("area_late")], axis=1).reset_index()
    tab["ratio"] = tab["area_late"] / tab["area_early"].replace(0, np.nan)
    tab["expansion"] = (tab["ratio"] > 1.5) & (tab["area_late"] >= 50_000)
    tab["stable_major"] = (tab["area_early"] >= tab["area_early"].quantile(0.80)) & (tab["ratio"].between(0.67, 1.5))
    write_table(tab, OUT / "area_stability.csv")
    return tab


def cropmask_sensitivity(food, trade, labels, obs_yield, obs_prod, area_tab) -> dict:
    expanders = set(zip(area_tab.loc[area_tab["expansion"], "iso3"], area_tab.loc[area_tab["expansion"], "crop"]))
    stable = set(zip(area_tab.loc[area_tab["stable_major"], "iso3"], area_tab.loc[area_tab["stable_major"], "crop"]))

    def _zero(frame, keys):
        out = frame.copy()
        mask = [((r.iso3, r.crop) in keys) for r in out.itertuples(index=False)]
        out.loc[mask, "pred"] = 0.0
        if "shock_hat" in out.columns:
            out.loc[mask, "shock_hat"] = 0.0
        return out

    def _keep(frame, keys):
        return frame[[((r.iso3, r.crop) in keys) for r in frame.itertuples(index=False)]].copy()

    recs = {}
    for tag, obs in (("yield", None), ("production", obs_prod)):
        suite_full, _ = None, None
        for name, panel in (
            ("zero_expansion_forecast", _zero(food, expanders)),
            ("stable_majors_only", food),
        ):
            use = panel
            if name == "stable_majors_only" and stable:
                # zero everyone who is not a long-standing major producer
                others = set(zip(food["iso3"], food["crop"])) - stable
                use = _zero(food, others)
            suite = evaluate_exposure_suite(use, trade, labels, label_mode="pooled", obs_shocks=obs)
            s192 = suite[suite["lead"] == 192] if len(suite) else suite
            recs[f"{tag}_{name}"] = {
                "auprc": float(s192["auprc_forecast"].mean()) if len(s192) else np.nan,
                "n_year_crop": int(len(s192)),
            }
    write_table(pd.DataFrame([{"test": k, **v} for k, v in recs.items()]), OUT / "cropmask_sensitivity.csv")
    return recs


def extended_rolling_origin(labels, series, trade, obs_prod) -> dict:
    cfg = load_config(ROOT)
    panel = labels[labels["crop"].isin(CROPS)].copy()
    series = series[series["crop"].isin(CROPS)].copy()
    feat = build_feature_table(panel, series, [192, 0], PhenologyVersion.DYNAMIC, ["optical", "climate"])
    rows = []
    pred_rows = []
    for lead, block in feat.groupby("lead"):
        block = block[np.isfinite(block["yield_anomaly"])].reset_index(drop=True)
        for split in rolling_origin(block, 2010, 8):
            pred = fit_predict("xgb", block.loc[split.train_idx], block.loc[split.test_idx], "yield_anomaly", True, cfg)
            y = block.loc[split.test_idx, "yield_anomaly"].to_numpy(dtype=float)
            rec = regression_metrics(y, pred)
            rec.update({"lead": int(lead), "year": int(split.test_year)})
            rows.append(rec)
            te = block.loc[split.test_idx, ["iso3", "crop", "year"]].copy()
            te["lead"] = int(lead)
            te["pred"] = pred
            te["shock_true"] = y
            pred_rows.append(te)
    skill = pd.DataFrame(rows)
    write_table(skill, OUT / "extended_ro_yield_skill.csv")
    preds = pd.concat(pred_rows, ignore_index=True)
    write_table(preds, OUT / "extended_ro_predictions.csv")
    food = preds.copy()
    out = {}
    for start in (2010, 2012, 2016):
        sub = food[food["year"] >= start]
        for tag, obs in (("yield", None), ("production", obs_prod)):
            suite = evaluate_exposure_suite(sub, trade, labels, label_mode="pooled", obs_shocks=obs)
            s192 = suite[suite["lead"] == 192]
            out[f"{tag}_from_{start}"] = {
                "auprc": float(s192["auprc_forecast"].mean()) if len(s192) else np.nan,
                "n0": float(s192["auprc_n0_lag"].mean()) if len(s192) else np.nan,
                "n_years": int(s192["year"].nunique()) if len(s192) else 0,
            }
    write_table(pd.DataFrame([{"test": k, **v} for k, v in out.items()]), OUT / "extended_ro_exposure.csv")
    return out


def cascade_ci(yield_suite, prod_suite, short_year: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = yield_suite[yield_suite["lead"] == 192][["year", "crop", "auprc_forecast", "prevalence"]].copy()
    y["stage"] = "yield_induced"
    y["auprc"] = y["auprc_forecast"]
    p = prod_suite[prod_suite["lead"] == 192][["year", "crop", "auprc_forecast", "prevalence"]].copy()
    p["stage"] = "production"
    p["auprc"] = p["auprc_forecast"]
    s = short_year[short_year["definition"] == "yoy"][["year", "crop", "auprc_q20", "pi_q20"]].copy()
    s = s.rename(columns={"auprc_q20": "auprc", "pi_q20": "prevalence"})
    s["stage"] = "shortfall"
    long = pd.concat(
        [y[["year", "crop", "stage", "auprc", "prevalence"]], p[["year", "crop", "stage", "auprc", "prevalence"]], s],
        ignore_index=True,
    )
    long["skill"] = [skill_score(a, pr) for a, pr in zip(long["auprc"], long["prevalence"])]
    write_table(long, OUT / "cascade_year_crop.csv")
    rng = np.random.default_rng(42)
    recs = []
    for stage in ("yield_induced", "production", "shortfall"):
        sub = long[long["stage"] == stage]
        by_year = sub.groupby("year")["skill"].mean()
        draws = [float(np.mean(rng.choice(by_year.to_numpy(), size=len(by_year), replace=True))) for _ in range(N_BOOT)]
        ds = pd.Series(draws)
        recs.append(
            {
                "stage": stage,
                "skill": float(by_year.mean()),
                "auprc": float(sub.groupby("year")["auprc"].mean().mean()),
                "prevalence": float(sub.groupby("year")["prevalence"].mean().mean()),
                "p50": float(ds.median()),
                "p025": float(ds.quantile(0.025)),
                "p975": float(ds.quantile(0.975)),
                "p05": float(ds.quantile(0.05)),
                "p95": float(ds.quantile(0.95)),
            }
        )
    ymap = {st: long[long["stage"] == st].groupby("year")["skill"].mean() for st in ("yield_induced", "production", "shortfall")}
    pairs = []
    for a, b in (("yield_induced", "production"), ("production", "shortfall"), ("yield_induced", "shortfall")):
        common = sorted(set(ymap[a].index) & set(ymap[b].index))
        diffs = []
        for _ in range(N_BOOT):
            pick = rng.choice(common, size=len(common), replace=True)
            diffs.append(float(np.mean([ymap[a].loc[y] - ymap[b].loc[y] for y in pick])))
        ds = pd.Series(diffs)
        pairs.append(
            {
                "contrast": f"{a}-{b}",
                "delta": float(np.mean([ymap[a].loc[y] - ymap[b].loc[y] for y in common])),
                "p025": float(ds.quantile(0.025)),
                "p975": float(ds.quantile(0.975)),
                "p05": float(ds.quantile(0.05)),
                "p95": float(ds.quantile(0.95)),
            }
        )
    write_table(pd.DataFrame(recs), OUT / "cascade_skill_ci.csv")
    write_table(pd.DataFrame(pairs), OUT / "cascade_pairwise.csv")
    return pd.DataFrame(recs), pd.DataFrame(pairs)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preds = read_table(ROOT / "results" / "predictions.csv")
    labels = read_table(ROOT / "data_interim" / "obs_panel_labels.csv")
    trade = read_table(ROOT / "data_interim" / "obs_trade_network.csv")
    series = read_table(ROOT / "data_interim" / "obs_season_series.csv")
    food = preds[(preds["experiment"] == "P_food_system") & preds["crop"].isin(CROPS)].copy()
    labels = labels[labels["crop"].isin(CROPS)].copy()
    trade = trade[trade["crop"].isin(CROPS)].copy()
    series = series[series["crop"].isin(CROPS)].copy()
    obs_prod = production_obs_shocks(labels)

    summary: dict = {}
    print("yield-induced suite", flush=True)
    ysuite, ysum = run_endpoint_suite(food, trade, labels, None, "yield_lag1")
    print(json.dumps(ysum, indent=2, default=str), flush=True)
    print("production suite", flush=True)
    psuite, psum = run_endpoint_suite(food, trade, labels, obs_prod, "production_lag1")
    print(json.dumps(psum, indent=2, default=str), flush=True)
    summary["yield_lag1"] = ysum
    summary["production_lag1"] = psum

    print("t-2 network", flush=True)
    _, y2 = run_endpoint_suite(food, trade, labels, None, "yield_lag2", lag_years=2, do_perm=False, do_ablate=False)
    _, p2 = run_endpoint_suite(food, trade, labels, obs_prod, "production_lag2", lag_years=2, do_perm=False, do_ablate=False)
    summary["yield_lag2"] = y2
    summary["production_lag2"] = p2
    print("t-2", json.dumps({"y": y2.get("auprc_192"), "p": p2.get("auprc_192")}, default=str), flush=True)

    print("shortfall sensitivity", flush=True)
    short = shortfall_sensitivity(food, trade, labels)
    summary["shortfall"] = short.to_dict("records") if len(short) else []
    print(short.to_string(index=False), flush=True)

    print("crop-mask / expansion", flush=True)
    area_tab = expansion_exporters(labels)
    summary["cropmask"] = cropmask_sensitivity(food, trade, labels, None, obs_prod, area_tab)
    print(json.dumps(summary["cropmask"], indent=2, default=str), flush=True)

    print("extended rolling-origin", flush=True)
    summary["extended_ro"] = extended_rolling_origin(labels, series, trade, obs_prod)
    print(json.dumps(summary["extended_ro"], indent=2, default=str), flush=True)

    short_year = read_table(OUT / "shortfall_sensitivity_year.csv")
    casc, pairs = cascade_ci(ysuite, psuite, short_year)
    summary["cascade"] = casc.to_dict("records")
    summary["pairwise"] = pairs.to_dict("records")
    print(casc.to_string(index=False), flush=True)
    print(pairs.to_string(index=False), flush=True)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
