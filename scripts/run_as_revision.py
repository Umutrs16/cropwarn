"""Agricultural Systems P0: formula-consistent S, paired ΔAP, Top-K.

Uses existing official predictions and revision_p6 suites. Writes only to
results/as_revision/. Does not overwrite results/predictions.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.data.io import read_table, write_table
from cropwarn.data.labels import rolling_trend_anomaly
from cropwarn.trade.baselines import importer_specific_truth, network_baseline_scores, _labels_for_propensity
from cropwarn.trade.exposure import as_supply_shocks, import_exposure, observed_exposure_history, overlay_observed_shocks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "as_revision"
P6 = ROOT / "results" / "revision_p6"
LEAD = 192
N_BOOT = 2000
KS = (5, 8, 10, 15, 20)


def skill(ap: float, pi: float) -> float:
    if not np.isfinite(ap) or not np.isfinite(pi) or pi >= 1.0:
        return float("nan")
    return float((ap - pi) / (1.0 - pi))


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


def year_block_stat(values: pd.Series, n_boot: int = N_BOOT, seed: int = 42) -> dict:
    arr = pd.to_numeric(values, errors="coerce").dropna()
    if arr.empty:
        return {"mean": np.nan, "p025": np.nan, "p975": np.nan}
    rng = np.random.default_rng(seed)
    draws = [float(np.mean(rng.choice(arr.to_numpy(), size=len(arr), replace=True))) for _ in range(n_boot)]
    s = pd.Series(draws)
    return {"mean": float(arr.mean()), "p025": float(s.quantile(0.025)), "p975": float(s.quantile(0.975))}


def formula_s_from_suite(suite: pd.DataFrame, lead: int = LEAD, n_boot: int = N_BOOT) -> dict:
    sub = suite[suite["lead"] == int(lead)].copy()
    sub["best_net"] = sub[["auprc_n0_lag", "auprc_n1_mean", "auprc_n2_hhi", "auprc_n3_propensity"]].max(axis=1)
    year = sub.groupby("year", as_index=False).agg(
        ap=("auprc_forecast", "mean"),
        n0=("auprc_n0_lag", "mean"),
        best_net=("best_net", "mean"),
        pi=("prevalence", "mean"),
    )
    year["s"] = [skill(a, p) for a, p in zip(year["ap"], year["pi"])]
    year["d_n0"] = year["ap"] - year["n0"]
    year["d_best"] = year["ap"] - year["best_net"]
    ap = float(year["ap"].mean())
    pi = float(year["pi"].mean())
    s_formula = skill(ap, pi)
    rng = np.random.default_rng(42)
    s_draws, ap_draws, d0_draws, db_draws = [], [], [], []
    yrs = year["year"].to_numpy()
    for _ in range(n_boot):
        pick = rng.choice(len(yrs), size=len(yrs), replace=True)
        block = year.iloc[pick]
        a = float(block["ap"].mean())
        p = float(block["pi"].mean())
        ap_draws.append(a)
        s_draws.append(skill(a, p))
        d0_draws.append(float(block["d_n0"].mean()))
        db_draws.append(float(block["d_best"].mean()))
    return {
        "n_years": int(len(year)),
        "n_year_crop": int(len(sub)),
        "auprc": ap,
        "prevalence": pi,
        "s_formula": s_formula,
        "s_year_mean": float(year["s"].mean()),
        "auprc_p025": float(np.quantile(ap_draws, 0.025)),
        "auprc_p975": float(np.quantile(ap_draws, 0.975)),
        "s_p025": float(np.quantile(s_draws, 0.025)),
        "s_p975": float(np.quantile(s_draws, 0.975)),
        "delta_n0": float(year["d_n0"].mean()),
        "delta_n0_p025": float(np.quantile(d0_draws, 0.025)),
        "delta_n0_p975": float(np.quantile(d0_draws, 0.975)),
        "delta_best": float(year["d_best"].mean()),
        "delta_best_p025": float(np.quantile(db_draws, 0.025)),
        "delta_best_p975": float(np.quantile(db_draws, 0.975)),
        "n0": float(year["n0"].mean()),
        "best_net": float(year["best_net"].mean()),
        "formula_holds": bool(abs(s_formula - skill(ap, pi)) < 1e-12),
        "by_year": year.to_dict("records"),
    }


def build_long_192(
    food: pd.DataFrame,
    trade: pd.DataFrame,
    labels: pd.DataFrame,
    obs_shocks: pd.DataFrame | None,
    lag_years: int = 1,
) -> pd.DataFrame:
    history = observed_exposure_history(labels, trade, lag_years, obs_shocks=obs_shocks)
    shocks = as_supply_shocks(food[food["lead"] == LEAD], labels)
    if obs_shocks is not None:
        shocks = overlay_observed_shocks(shocks, obs_shocks)
    labels_n3 = _labels_for_propensity(labels, obs_shocks)
    rows = []
    for (year, crop), g in shocks.groupby(["year", "crop"]):
        need = [c for c in ("iso3", "year", "crop", "lead", "shock_hat", "shock_true") if c in g.columns]
        expo = import_exposure(g[need], trade, int(year), str(crop), lag_years)
        if expo.empty:
            continue
        expo = expo.rename(columns={"exposure": "exposure_hat", "flow_true": "exposure_obs"}).reset_index(drop=True)
        past = history[(history["year"] < int(year)) & (history["crop"] == crop)] if len(history) else pd.DataFrame()
        truth = importer_specific_truth(expo, past, q=0.80, mode="pooled")
        base = network_baseline_scores(expo, history, trade, int(year), str(crop), lag_years, labels=labels_n3)
        g2 = expo.merge(base, on="importer", how="left")
        g2["truth"] = truth.to_numpy()
        g2["year"] = int(year)
        g2["crop"] = crop
        g2["lead"] = LEAD
        g2["lag_years"] = lag_years
        rows.append(g2)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def topk_table(long: pd.DataFrame, score_col: str, ks=KS) -> pd.DataFrame:
    recs = []
    for (year, crop), g in long.groupby(["year", "crop"]):
        g = g.copy()
        g["score"] = pd.to_numeric(g[score_col], errors="coerce")
        g = g.dropna(subset=["score"]).sort_values("score", ascending=False)
        truth = g["truth"].astype(bool)
        obs = pd.to_numeric(g["exposure_obs"], errors="coerce").fillna(0.0)
        n_high = int(truth.sum())
        tot_obs = float(obs.sum())
        tot_high_obs = float(obs[truth].sum())
        for k in ks:
            top = g.head(int(k))
            hit = top["truth"].astype(bool)
            recs.append(
                {
                    "year": int(year),
                    "crop": crop,
                    "score": score_col,
                    "k": int(k),
                    "precision": float(hit.mean()) if len(top) else np.nan,
                    "recall": float(hit.sum() / n_high) if n_high else np.nan,
                    "captured_obs_share": float(pd.to_numeric(top["exposure_obs"], errors="coerce").fillna(0).sum() / tot_obs)
                    if tot_obs > 0
                    else np.nan,
                    "captured_high_share": float(pd.to_numeric(top.loc[hit, "exposure_obs"], errors="coerce").fillna(0).sum() / tot_high_obs)
                    if tot_high_obs > 0
                    else np.nan,
                    "n_high": n_high,
                    "n": int(len(g)),
                }
            )
    return pd.DataFrame(recs)


def summarize_topk(tk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (score, k), g in tk.groupby(["score", "k"]):
        prec = year_block_stat(g.groupby("year")["precision"].mean())
        rec = year_block_stat(g.groupby("year")["recall"].mean())
        cap = year_block_stat(g.groupby("year")["captured_obs_share"].mean())
        rows.append(
            {
                "score": score,
                "k": int(k),
                "precision": prec["mean"],
                "precision_p025": prec["p025"],
                "precision_p975": prec["p975"],
                "recall": rec["mean"],
                "recall_p025": rec["p025"],
                "recall_p975": rec["p975"],
                "captured_obs_share": cap["mean"],
                "captured_p025": cap["p025"],
                "captured_p975": cap["p975"],
            }
        )
    return pd.DataFrame(rows).sort_values(["score", "k"])


def yield_by_stratum(preds: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    p = preds[(preds["experiment"] == "P_food_system") & preds["crop"].isin(["wheat", "soybean"])].copy()
    extra = labels[labels["crop"].isin(["wheat", "soybean"])][["iso3", "crop", "year", "irrigation"]].drop_duplicates()
    p = p.merge(extra, on=["iso3", "crop", "year"], how="left")
    p["irrigation_bin"] = np.where(pd.to_numeric(p["irrigation"], errors="coerce") >= 0.20, "higher_irrigation", "lower_irrigation")
    yhat = pd.to_numeric(p["pred"], errors="coerce")
    y = pd.to_numeric(p["yield_anomaly"], errors="coerce")
    p["yhat"] = yhat
    p["y"] = y
    recs = []
    for lead, g in p.groupby("lead"):
        for col, name in (("latitude_band", "climate_band"), ("irrigation_bin", "irrigation"), ("crop", "crop")):
            for key, h in g.groupby(col):
                m = h["yhat"].notna() & h["y"].notna()
                if int(m.sum()) < 12:
                    continue
                yy = h.loc[m, "y"].to_numpy()
                yh = h.loc[m, "yhat"].to_numpy()
                ss_res = float(np.sum((yy - yh) ** 2))
                ss_tot = float(np.sum((yy - yy.mean()) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
                recs.append({"lead": int(lead), "stratum": name, "level": str(key), "n": int(m.sum()), "r2": r2})
    return pd.DataFrame(recs)


def sample_table(labels: pd.DataFrame, trade: pd.DataFrame) -> pd.DataFrame:
    lab = labels[labels["crop"].isin(["wheat", "soybean"])].copy()
    rows = []
    for crop, g in lab.groupby("crop"):
        rows.append(
            {
                "crop": crop,
                "n_countries": int(g["iso3"].nunique()),
                "n_country_years": int(g[["iso3", "year"]].drop_duplicates().shape[0]),
                "year_start": int(g["year"].min()),
                "year_end": int(g["year"].max()),
                "mean_harvested_mha": float(g.groupby("year")["harvested_ha"].sum().mean() / 1e6),
                "temperate": int(g.loc[g["latitude_band"] == "temperate", "iso3"].nunique()),
                "subtropical": int(g.loc[g["latitude_band"] == "subtropical", "iso3"].nunique()),
                "tropical": int(g.loc[g["latitude_band"] == "tropical", "iso3"].nunique()),
            }
        )
    t = trade[trade["crop"].isin(["wheat", "soybean"])]
    rows.append(
        {
            "crop": "trade_importers",
            "n_countries": int(t["importer"].nunique()),
            "n_country_years": int(t[["importer", "year", "crop"]].drop_duplicates().shape[0]),
            "year_start": int(t["year"].min()),
            "year_end": int(t["year"].max()),
            "mean_harvested_mha": np.nan,
            "temperate": np.nan,
            "subtropical": np.nan,
            "tropical": np.nan,
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ysuite = read_table(P6 / "suite_yield_lag1.csv")
    psuite = read_table(P6 / "suite_production_lag1.csv")
    y2 = read_table(P6 / "suite_yield_lag2.csv")
    p2 = read_table(P6 / "suite_production_lag2.csv")
    y_s = formula_s_from_suite(ysuite)
    p_s = formula_s_from_suite(psuite)
    y2_s = formula_s_from_suite(y2)
    p2_s = formula_s_from_suite(p2)
    write_table(pd.DataFrame(y_s["by_year"]).assign(endpoint="yield_induced"), OUT / "year_skill_yield.csv")
    write_table(pd.DataFrame(p_s["by_year"]).assign(endpoint="production"), OUT / "year_skill_production.csv")

    preds = read_table(ROOT / "results" / "predictions.csv")
    labels = read_table(ROOT / "data_interim" / "obs_panel_labels.csv")
    trade = read_table(ROOT / "data_interim" / "obs_trade_network.csv")
    food = preds[(preds["experiment"] == "P_food_system") & preds["crop"].isin(["wheat", "soybean"])].copy()
    labels = labels[labels["crop"].isin(["wheat", "soybean"])].copy()
    trade = trade[trade["crop"].isin(["wheat", "soybean"])].copy()
    prod_shocks = production_obs_shocks(labels)

    long_y = build_long_192(food, trade, labels, None, 1)
    long_p = build_long_192(food, trade, labels, prod_shocks, 1)
    write_table(long_y, OUT / "rows_yield_192.csv")
    write_table(long_p, OUT / "rows_production_192.csv")

    tk_parts = []
    for long, tag in ((long_p, "production"), (long_y, "yield")):
        for col in ("exposure_hat", "n0_lag", "n3_propensity"):
            if col not in long.columns:
                continue
            part = topk_table(long, col)
            part["endpoint"] = tag
            tk_parts.append(part)
    tk = pd.concat(tk_parts, ignore_index=True)
    write_table(tk, OUT / "topk_year_crop.csv")
    summ_rows = []
    for endpoint, g in tk.groupby("endpoint"):
        s = summarize_topk(g)
        s["endpoint"] = endpoint
        summ_rows.append(s)
    tk_sum = pd.concat(summ_rows, ignore_index=True)
    write_table(tk_sum, OUT / "topk_summary.csv")

    # Paired Top-K increment: forecast − N0 on the same year×crop
    paired_k = []
    for endpoint, g in tk.groupby("endpoint"):
        wide = g.pivot_table(index=["year", "crop", "k"], columns="score", values="precision")
        if "exposure_hat" in wide.columns and "n0_lag" in wide.columns:
            d = (wide["exposure_hat"] - wide["n0_lag"]).reset_index()
            for k, h in d.groupby("k"):
                ci = year_block_stat(h.groupby("year")[0].mean() if 0 in h.columns else h.groupby("year").mean().iloc[:, 0])
                # after reset the value column may be named 0
                valcol = [c for c in h.columns if c not in ("year", "crop", "k")][0]
                ci = year_block_stat(h.groupby("year")[valcol].mean())
                paired_k.append({"endpoint": endpoint, "k": int(k), "delta_precision_vs_n0": ci["mean"], **{f"p{a}": ci[b] for a, b in (("025", "p025"), ("975", "p975"))}})
    write_table(pd.DataFrame(paired_k), OUT / "topk_paired_delta.csv")

    strata = yield_by_stratum(preds, labels)
    write_table(strata, OUT / "yield_skill_by_stratum.csv")
    sample = sample_table(labels, trade)
    write_table(sample, OUT / "sample_coverage.csv")

    # Micro-pooled AP on the same long table (sensitivity, not headline)
    def _micro(long: pd.DataFrame, score: str) -> dict:
        y = long["truth"].astype(int)
        s = pd.to_numeric(long[score], errors="coerce")
        m = y.notna() & s.notna()
        ap = float(average_precision_score(y[m], s[m])) if int(y[m].nunique()) > 1 else np.nan
        pi = float(y[m].mean())
        return {"micro_ap": ap, "micro_pi": pi, "micro_s": skill(ap, pi), "n": int(m.sum())}

    summary = {
        "note": "Headline AP is the year-block mean of year×crop AUPRC (official contract). Headline S uses S=(AP−π)/(1−π) on those same two means. The previous 0.53/0.40 values were means of yearly S and do not satisfy the formula.",
        "yield_induced": {k: v for k, v in y_s.items() if k != "by_year"},
        "production": {k: v for k, v in p_s.items() if k != "by_year"},
        "yield_t2": {k: v for k, v in y2_s.items() if k != "by_year"},
        "production_t2": {k: v for k, v in p2_s.items() if k != "by_year"},
        "micro_yield": _micro(long_y, "exposure_hat"),
        "micro_production": _micro(long_p, "exposure_hat"),
        "micro_yield_n0": _micro(long_y, "n0_lag"),
        "micro_production_n0": _micro(long_p, "n0_lag"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("yield_induced", "production", "yield_t2", "production_t2")}, indent=2))
    print("topk", tk_sum.to_string(index=False))
    print("ok", OUT)


if __name__ == "__main__":
    main()
