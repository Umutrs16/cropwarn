"""Production-shock to import-exposure mapping (Nature Food hook)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cropwarn.validation.leakage import assert_no_future_trade


def import_exposure(
    shocks: pd.DataFrame,
    trade: pd.DataFrame,
    forecast_year: int,
    crop: str,
    lag_years: int = 1,
) -> pd.DataFrame:
    """Exposure(j,c,t,L) = sum_i W(i->j,c) * max(0, -Shock_i)."""

    trade_year = forecast_year - lag_years
    assert_no_future_trade(trade_year, forecast_year)
    w = trade[(trade["year"] == trade_year) & (trade["crop"] == crop)].copy()
    s = shocks[(shocks["year"] == forecast_year) & (shocks["crop"] == crop)].copy()
    if w.empty or s.empty:
        return pd.DataFrame()
    s = s.rename(columns={"iso3": "exporter"})
    if "harvested_ha" in s.columns:
        rows = []
        for (exp, lead), g in s.groupby(["exporter", "lead"]):
            w = pd.to_numeric(g["harvested_ha"], errors="coerce").fillna(1.0).clip(lower=1.0)
            rows.append(
                {
                    "exporter": exp,
                    "lead": lead,
                    "shock_hat": float(np.average(g["shock_hat"].to_numpy(dtype=float), weights=w)),
                    "shock_true": float(np.average(g["shock_true"].to_numpy(dtype=float), weights=w)),
                }
            )
        s = pd.DataFrame(rows)
    else:
        s = (
            s.groupby(["exporter", "lead"], as_index=False)[["shock_hat", "shock_true"]]
            .mean()
        )
    merged = w.merge(s, on="exporter", how="left")
    merged["neg_shock"] = np.clip(-merged["shock_hat"].astype(float), 0.0, None)
    merged["flow_exposure"] = merged["export_share"] * merged["neg_shock"]
    expo = (
        merged.groupby(["importer", "crop", "lead"], as_index=False)
        .agg(
            exposure=("flow_exposure", "sum"),
            import_dependency=("import_dependency", "mean"),
        )
    )
    true = w.merge(s[["exporter", "lead", "shock_true"]], on="exporter", how="left")
    true["neg_true"] = np.clip(-true["shock_true"].astype(float), 0.0, None)
    true["flow_true"] = true["export_share"] * true["neg_true"]
    truth = true.groupby(["importer", "lead"], as_index=False)["flow_true"].sum()
    expo = expo.merge(truth, on=["importer", "lead"], how="left")
    expo["year"] = forecast_year
    expo["weighted_exposure"] = expo["exposure"] * expo["import_dependency"]
    return expo


def high_risk_detection(
    expo: pd.DataFrame,
    q: float = 0.80,
) -> dict[str, float]:
    if expo.empty:
        return {
            "recall": np.nan,
            "false_alarm": np.nan,
            "precision": np.nan,
            "f1": np.nan,
            "auprc": np.nan,
            "auroc": np.nan,
            "n_high": 0.0,
        }
    truth_flag = expo["flow_true"] >= expo["flow_true"].quantile(q)
    pred_flag = expo["exposure"] >= expo["exposure"].quantile(q)
    tp = float((truth_flag & pred_flag).sum())
    fn = float((truth_flag & ~pred_flag).sum())
    fp = float((~truth_flag & pred_flag).sum())
    tn = float((~truth_flag & ~pred_flag).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    auprc = float("nan")
    auroc = float("nan")
    y = truth_flag.to_numpy(dtype=int)
    s = pd.to_numeric(expo["exposure"], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(s)
    if m.sum() >= 8 and len(np.unique(y[m])) == 2:
        from sklearn.metrics import average_precision_score, roc_auc_score

        try:
            auprc = float(average_precision_score(y[m], s[m]))
        except ValueError:
            auprc = float("nan")
        try:
            auroc = float(roc_auc_score(y[m], s[m]))
        except ValueError:
            auroc = float("nan")
    return {
        "recall": recall,
        "false_alarm": fp / max(fp + tn, 1.0),
        "precision": precision,
        "f1": f1,
        "auprc": auprc,
        "auroc": auroc,
        "n_high": float(truth_flag.sum()),
    }


def evaluate_trade(
    pred_panel: pd.DataFrame,
    trade: pd.DataFrame,
    lag_years: int,
    high_q: float,
) -> pd.DataFrame:
    rows = []
    for (year, crop, lead), g in pred_panel.groupby(["year", "crop", "lead"]):
        shocks = g.rename(columns={"pred": "shock_hat"})[
            ["iso3", "year", "crop", "lead", "shock_hat", "shock_true"]
        ]
        expo = import_exposure(shocks, trade, int(year), str(crop), lag_years)
        if expo.empty:
            continue
        det = high_risk_detection(expo, high_q)
        rows.append(
            {
                "year": int(year),
                "crop": crop,
                "lead": int(lead),
                **det,
                "mean_exposure": float(expo["exposure"].mean()),
            }
        )
    return pd.DataFrame(rows)


def lagged_area_production(labels: pd.DataFrame) -> pd.DataFrame:
    """Year-t lookup of area and production from year t−1 (no contemporaneous leak)."""

    src = labels.copy()
    if "production_t" not in src.columns and {"yield_tha", "harvested_ha"} <= set(src.columns):
        src["production_t"] = pd.to_numeric(src["yield_tha"], errors="coerce") * pd.to_numeric(
            src["harvested_ha"], errors="coerce"
        )
    keys = [c for c in ("iso3", "crop") if c in src.columns]
    cols = keys + [c for c in ("year", "harvested_ha", "production_t", "yield_tha") if c in src.columns]
    lag = src[cols].drop_duplicates().copy()
    lag = lag.rename(columns={"harvested_ha": "area_lag1", "production_t": "prod_lag1", "yield_tha": "yield_lag1"})
    lag["year"] = lag["year"] + 1
    return lag


def as_supply_shocks(pred_panel: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Δyield × area_{t−1} / production_{t−1} — prospective relative supply shock."""

    out = pred_panel.copy()
    lag = lagged_area_production(labels)
    keys = [c for c in ("iso3", "crop", "year") if c in out.columns and c in lag.columns]
    out = out.merge(lag, on=keys, how="left")
    prod = pd.to_numeric(out.get("prod_lag1"), errors="coerce")
    area = pd.to_numeric(out.get("area_lag1"), errors="coerce")
    if "pred" in out.columns:
        out["shock_hat"] = pd.to_numeric(out["pred"], errors="coerce") * area / prod.replace(0, np.nan)
    if "shock_true" in out.columns:
        out["shock_true"] = pd.to_numeric(out["shock_true"], errors="coerce") * area / prod.replace(0, np.nan)
    return out


def import_quantity_panel(trade: pd.DataFrame) -> pd.DataFrame:
    if trade is None or trade.empty or "qty" not in trade.columns:
        return pd.DataFrame()
    return trade.groupby(["year", "crop", "importer"], as_index=False)["qty"].sum()


def import_rolling_anomaly(trade: pd.DataFrame, min_history: int = 4) -> pd.DataFrame:
    """Causal rolling percent deviation of importer quantity from a t−1-only trend."""

    from cropwarn.data.labels import rolling_trend_anomaly

    imp = import_quantity_panel(trade)
    if imp.empty:
        return pd.DataFrame()
    tmp = imp.rename(columns={"importer": "iso3", "qty": "yield_tha"})
    anom = rolling_trend_anomaly(tmp, value_col="yield_tha", method="percent", min_history=min_history)
    out = anom.rename(columns={"iso3": "importer", "yield_anomaly": "import_anomaly", "yield_tha": "qty"})
    keep = [c for c in ("year", "crop", "importer", "qty", "import_anomaly") if c in out.columns]
    return out[keep]


def realized_import_change(trade: pd.DataFrame) -> pd.DataFrame:
    if trade is None or trade.empty or "qty" not in trade.columns:
        return pd.DataFrame()
    imp = trade.groupby(["year", "crop", "importer"], as_index=False)["qty"].sum()
    imp = imp.sort_values(["importer", "crop", "year"])
    imp["qty_lag1"] = imp.groupby(["importer", "crop"])["qty"].shift(1)
    imp["import_chg"] = imp["qty"] / imp["qty_lag1"].replace(0, np.nan) - 1.0
    return imp


def evaluate_realized_import_shortfall(
    pred_panel: pd.DataFrame,
    trade: pd.DataFrame,
    lag_years: int,
    high_q: float = 0.80,
    labels: pd.DataFrame | None = None,
    shortfall_q: float = 0.20,
) -> pd.DataFrame:
    """Rank predicted exposure against observed importer quantity declines.

    Truth is a realized import shortfall (bottom shortfall_q of year-on-year qty change),
    not the model's own lagged-network exposure. This is weaker than price/stock
    validation but is an actual-outcome test.
    """

    chg = realized_import_change(trade)
    if chg.empty:
        return pd.DataFrame()
    if labels is not None and "harvested_ha" in labels.columns:
        shocks = as_supply_shocks(pred_panel, labels)
    else:
        shocks = pred_panel.copy()
        if "shock_hat" not in shocks.columns and "pred" in shocks.columns:
            shocks["shock_hat"] = shocks["pred"]
    rows = []
    for (year, crop, lead), g in shocks.groupby(["year", "crop", "lead"]):
        need = ["iso3", "year", "crop", "lead", "shock_hat", "shock_true"]
        if "shock_hat" not in g.columns and "pred" in g.columns:
            g = g.assign(shock_hat=g["pred"])
        keep = [c for c in need if c in g.columns]
        expo = import_exposure(g[keep], trade, int(year), str(crop), lag_years)
        if expo.empty:
            continue
        obs = chg[(chg["year"] == int(year)) & (chg["crop"] == crop)][["importer", "import_chg"]]
        expo = expo.merge(obs, on="importer", how="left")
        y = pd.to_numeric(expo["import_chg"], errors="coerce")
        if y.notna().sum() < 8:
            continue
        truth = y <= y.quantile(shortfall_q)
        det = high_risk_detection(
            expo.assign(flow_true=(-y).where(y.notna(), np.nan)),
            high_q,
        )
        # Override truth with realized shortfall rather than model-implied exposure
        pred_flag = expo["exposure"] >= expo["exposure"].quantile(high_q)
        tp = float((truth & pred_flag).sum())
        fn = float((truth & ~pred_flag).sum())
        fp = float((~truth & pred_flag).sum())
        prec = tp / max(tp + fp, 1.0)
        rec = tp / max(tp + fn, 1.0)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        auprc = det["auprc"]
        s = pd.to_numeric(expo["exposure"], errors="coerce")
        m = y.notna() & s.notna()
        if m.sum() >= 8 and truth[m].nunique() == 2:
            from sklearn.metrics import average_precision_score

            try:
                auprc = float(average_precision_score(truth[m].astype(int), s[m]))
            except ValueError:
                auprc = float("nan")
        rows.append(
            {
                "year": int(year),
                "crop": crop,
                "lead": int(lead),
                "recall": rec,
                "precision": prec,
                "f1": f1,
                "auprc": auprc,
                "n_importers": float(m.sum()),
                "n_shortfall": float(truth.sum()),
            }
        )
    return pd.DataFrame(rows)


def _continuous_exposure_skill(hat: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    m = np.isfinite(hat) & np.isfinite(obs)
    if m.sum() < 8:
        return {"n": float(m.sum()), "spearman": np.nan, "r2": np.nan}
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score

    rho = float("nan")
    try:
        rho = float(spearmanr(hat[m], obs[m]).statistic)
    except Exception:
        rho = float("nan")
    return {"n": float(m.sum()), "spearman": rho, "r2": float(r2_score(obs[m], hat[m]))}


def overlay_observed_shocks(shocks: pd.DataFrame, obs_shocks: pd.DataFrame) -> pd.DataFrame:
    """Replace shock_true with an alternative observed exporter shock; keep shock_hat."""

    if obs_shocks is None or obs_shocks.empty:
        return shocks
    out = shocks.copy()
    o = obs_shocks.rename(columns={"shock_obs": "shock_true_alt"})
    keys = [c for c in ("iso3", "crop", "year") if c in out.columns and c in o.columns]
    if "shock_true_alt" not in o.columns:
        return out
    out = out.merge(o[keys + ["shock_true_alt"]].drop_duplicates(), on=keys, how="left")
    out["shock_true"] = pd.to_numeric(out["shock_true_alt"], errors="coerce")
    return out.drop(columns=["shock_true_alt"])


def observed_exposure_history(
    labels: pd.DataFrame,
    trade: pd.DataFrame,
    lag_years: int = 1,
    obs_shocks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Ex-post exposure from observed shocks only (no forecast)."""

    hist = labels.copy()
    if "yield_anomaly" in hist.columns:
        hist["shock_true"] = pd.to_numeric(hist["yield_anomaly"], errors="coerce")
        hist["pred"] = hist["shock_true"]
    elif "shock_true" in hist.columns:
        hist["pred"] = hist["shock_true"]
    else:
        return pd.DataFrame()
    hist["lead"] = 0
    shocks = as_supply_shocks(hist, labels)
    if obs_shocks is not None:
        shocks = overlay_observed_shocks(shocks, obs_shocks)
    rows = []
    for (year, crop), g in shocks.groupby(["year", "crop"]):
        need = [c for c in ("iso3", "year", "crop", "lead", "shock_hat", "shock_true") if c in g.columns]
        expo = import_exposure(g[need], trade, int(year), str(crop), lag_years)
        if expo.empty:
            continue
        rows.append(expo.rename(columns={"flow_true": "exposure_obs"}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def importer_exposure_rows(
    pred_panel: pd.DataFrame,
    trade: pd.DataFrame,
    labels: pd.DataFrame,
    lag_years: int = 1,
) -> pd.DataFrame:
    """Importer-level predicted and observed exposure for every year–crop–lead."""

    shocks = as_supply_shocks(pred_panel, labels)
    if "shock_hat" not in shocks.columns and "pred" in shocks.columns:
        shocks = shocks.assign(shock_hat=shocks["pred"])
    pieces = []
    for (year, crop, lead), g in shocks.groupby(["year", "crop", "lead"]):
        need = [c for c in ("iso3", "year", "crop", "lead", "shock_hat", "shock_true") if c in g.columns]
        expo = import_exposure(g[need], trade, int(year), str(crop), lag_years)
        if expo.empty:
            continue
        pieces.append(expo.rename(columns={"exposure": "exposure_hat", "flow_true": "exposure_obs"}))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def evaluate_expost_exposure(
    pred_panel: pd.DataFrame,
    trade: pd.DataFrame,
    labels: pd.DataFrame,
    lag_years: int = 1,
    q: float = 0.80,
) -> pd.DataFrame:
    """Predicted exposure vs ex-post exposure from observed production shocks.

    Shock = Δyield × area_{t−1} / production_{t−1}.
    Exposure_i = sum_j share_{ij,t−1} × max(0, −Shock_j).
    High-exposure labels use Q_q of observed exposure in years < t,
    not a forced top 20% in every forecast year.
    """

    history = observed_exposure_history(labels, trade, lag_years)
    shocks = as_supply_shocks(pred_panel, labels)
    if "shock_hat" not in shocks.columns and "pred" in shocks.columns:
        shocks = shocks.assign(shock_hat=shocks["pred"])
    pieces = []
    for (year, crop, lead), g in shocks.groupby(["year", "crop", "lead"]):
        need = [c for c in ("iso3", "year", "crop", "lead", "shock_hat", "shock_true") if c in g.columns]
        expo = import_exposure(g[need], trade, int(year), str(crop), lag_years)
        if expo.empty:
            continue
        expo = expo.rename(columns={"exposure": "exposure_hat", "flow_true": "exposure_obs"})
        pieces.append(expo)
    if not pieces:
        return pd.DataFrame()
    all_expo = pd.concat(pieces, ignore_index=True)
    rows = []
    for (year, crop, lead), g in all_expo.groupby(["year", "crop", "lead"]):
        past = history[(history["year"] < int(year)) & (history["crop"] == crop)] if len(history) else pd.DataFrame()
        if past.empty or past["exposure_obs"].notna().sum() < 8:
            past = all_expo[(all_expo["year"] < int(year)) & (all_expo["crop"] == crop)]
        thresh = float(past["exposure_obs"].quantile(q)) if len(past) and past["exposure_obs"].notna().any() else float("nan")
        hat = pd.to_numeric(g["exposure_hat"], errors="coerce")
        obs = pd.to_numeric(g["exposure_obs"], errors="coerce")
        truth = obs > thresh if np.isfinite(thresh) else pd.Series(False, index=g.index)
        pred_flag = hat > thresh if np.isfinite(thresh) else pd.Series(False, index=g.index)
        tp = float((truth & pred_flag).sum())
        fn = float((truth & ~pred_flag).sum())
        fp = float((~truth & pred_flag).sum())
        prec = tp / max(tp + fp, 1.0)
        rec = tp / max(tp + fn, 1.0)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        auprc = float("nan")
        m = hat.notna() & obs.notna()
        if m.sum() >= 8 and int(truth[m].sum()) not in (0, int(m.sum())):
            from sklearn.metrics import average_precision_score

            try:
                auprc = float(average_precision_score(truth[m].astype(int), hat[m]))
            except ValueError:
                auprc = float("nan")
        cont = _continuous_exposure_skill(hat.to_numpy(dtype=float), obs.to_numpy(dtype=float))
        rows.append(
            {
                "year": int(year),
                "crop": crop,
                "lead": int(lead),
                "recall": rec,
                "precision": prec,
                "f1": f1,
                "auprc": auprc,
                "n_importers": float(m.sum()),
                "n_high": float(truth.sum()),
                "train_threshold": thresh,
                **{f"cont_{k}": v for k, v in cont.items()},
            }
        )
    return pd.DataFrame(rows)


def trade_early_warning_skill(auprc_lead: float, auprc_harvest: float, prevalence: float = 0.20) -> float:
    """Fraction of harvest-time AUPRC recovered at lead L, vs a prevalence baseline."""

    denom = auprc_harvest - prevalence
    if not np.isfinite(auprc_lead) or not np.isfinite(auprc_harvest) or abs(denom) < 1e-9:
        return float("nan")
    return float((auprc_lead - prevalence) / denom)
