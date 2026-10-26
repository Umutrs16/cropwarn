"""Network-only, importer-specific and permutation tests for trade exposure."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from cropwarn.trade.exposure import (
    _continuous_exposure_skill,
    as_supply_shocks,
    import_exposure,
    observed_exposure_history,
    overlay_observed_shocks,
)


def importer_specific_truth(
    g: pd.DataFrame,
    past: pd.DataFrame,
    q: float = 0.80,
    mode: str = "importer_q",
) -> pd.Series:
    """Per-importer historical threshold. mode: importer_q | importer_z | pooled."""

    obs = pd.to_numeric(g["exposure_obs"], errors="coerce")
    if mode == "pooled":
        thresh = float(past["exposure_obs"].quantile(q)) if len(past) and past["exposure_obs"].notna().any() else np.nan
        return obs > thresh if np.isfinite(thresh) else pd.Series(False, index=g.index)
    flags = []
    for idx, row in g.iterrows():
        hist = past[past["importer"] == row["importer"]]["exposure_obs"].dropna()
        val = float(obs.loc[idx]) if np.isfinite(obs.loc[idx]) else np.nan
        if len(hist) < 3 or not np.isfinite(val):
            flags.append(False)
            continue
        if mode == "importer_z":
            med = float(hist.median())
            mad = float(np.median(np.abs(hist - med))) * 1.4826
            flags.append(bool((val - med) / mad > 1.0) if mad > 1e-9 else False)
        else:
            flags.append(bool(val > float(hist.quantile(q))))
    return pd.Series(flags, index=g.index)


def network_baseline_scores(
    g: pd.DataFrame,
    history: pd.DataFrame,
    trade: pd.DataFrame,
    year: int,
    crop: str,
    lag_years: int = 1,
    labels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """N0 lagged exposure, N1 historical mean, N2 importer HHI, N3 supplier-shock propensity."""

    out = g[["importer"]].copy().reset_index(drop=True)
    past = history[(history["year"] < int(year)) & (history["crop"] == crop)] if len(history) else pd.DataFrame()
    lagged = history[(history["year"] == int(year) - 1) & (history["crop"] == crop)] if len(history) else pd.DataFrame()
    if len(lagged):
        out = out.merge(lagged[["importer", "exposure_obs"]].rename(columns={"exposure_obs": "n0_lag"}), on="importer", how="left")
    else:
        out["n0_lag"] = np.nan
    if len(past):
        mu = past.groupby("importer")["exposure_obs"].mean().rename("n1_mean")
        out = out.merge(mu, on="importer", how="left")
    else:
        out["n1_mean"] = np.nan
    w = trade[(trade["year"] == int(year) - lag_years) & (trade["crop"] == crop)] if trade is not None else pd.DataFrame()
    if len(w):
        hhi = (
            w.groupby("importer")["export_share"]
            .apply(lambda s: float(np.nansum(np.square(pd.to_numeric(s, errors="coerce")))))
            .rename("n2_hhi")
        )
        out = out.merge(hhi, on="importer", how="left")
    else:
        out["n2_hhi"] = np.nan
    out["n3_propensity"] = np.nan
    if labels is not None and len(w):
        hist = labels[(labels["year"] < int(year)) & (labels["crop"] == crop)].copy()
        if len(hist):
            dummy = hist.copy()
            dummy["lead"] = 0
            if "pred" not in dummy.columns:
                dummy["pred"] = dummy.get("yield_anomaly", dummy.get("shock_true"))
            if "shock_true" not in dummy.columns and "yield_anomaly" in dummy.columns:
                dummy["shock_true"] = dummy["yield_anomaly"]
            sh = as_supply_shocks(dummy, labels)
            mu = sh.groupby("iso3")["shock_true"].mean().rename("mean_shock")
            ww = w.merge(mu, left_on="exporter", right_index=True, how="left")
            ww["flow"] = pd.to_numeric(ww["export_share"], errors="coerce") * np.clip(
                -pd.to_numeric(ww["mean_shock"], errors="coerce"), 0.0, None
            )
            prop = ww.groupby("importer")["flow"].sum().rename("n3_propensity")
            out = out.drop(columns=["n3_propensity"], errors="ignore").merge(prop, on="importer", how="left")
    return out


def _auprc(truth: pd.Series, score: pd.Series) -> float:
    y = pd.Series(np.asarray(truth), dtype=float).reset_index(drop=True)
    s = pd.to_numeric(pd.Series(np.asarray(score)), errors="coerce").reset_index(drop=True)
    m = y.notna() & s.notna()
    if int(m.sum()) < 8 or int(y[m].nunique()) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y[m].astype(int), s[m]))
    except ValueError:
        return float("nan")


def _labels_for_propensity(labels: pd.DataFrame, obs_shocks: pd.DataFrame | None) -> pd.DataFrame:
    """N3 uses historical mean observed shocks; overlay when the label is not yield-induced."""

    if obs_shocks is None or obs_shocks.empty:
        return labels
    out = labels.copy()
    o = obs_shocks.rename(columns={"shock_obs": "alt"})
    keys = [c for c in ("iso3", "crop", "year") if c in out.columns and c in o.columns]
    out = out.merge(o[keys + ["alt"]].drop_duplicates(), on=keys, how="left")
    out["yield_anomaly"] = pd.to_numeric(out["alt"], errors="coerce")
    out["shock_true"] = out["yield_anomaly"]
    return out.drop(columns=["alt"])


def empirical_pvalue(draws: np.ndarray, observed: float) -> float:
    """P(perm ≥ observed) with +1 correction."""

    x = np.asarray(draws, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0 or not np.isfinite(observed):
        return float("nan")
    return float((1 + int(np.sum(x >= float(observed)))) / (len(x) + 1))


def skill_fraction(model: float, baseline: float, oracle: float) -> float:
    denom = float(oracle) - float(baseline)
    if not np.isfinite(denom) or abs(denom) < 1e-9:
        return float("nan")
    return float((float(model) - float(baseline)) / denom)


def evaluate_exposure_suite(
    pred_panel: pd.DataFrame,
    trade: pd.DataFrame,
    labels: pd.DataFrame,
    lag_years: int = 1,
    q: float = 0.80,
    label_mode: str = "importer_q",
    obs_shocks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    history = observed_exposure_history(labels, trade, lag_years, obs_shocks=obs_shocks)
    shocks = as_supply_shocks(pred_panel, labels)
    if "shock_hat" not in shocks.columns and "pred" in shocks.columns:
        shocks = shocks.assign(shock_hat=shocks["pred"])
    if obs_shocks is not None:
        shocks = overlay_observed_shocks(shocks, obs_shocks)
    labels_n3 = _labels_for_propensity(labels, obs_shocks)
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
        g = g.reset_index(drop=True)
        past = history[(history["year"] < int(year)) & (history["crop"] == crop)] if len(history) else pd.DataFrame()
        if past.empty:
            past = all_expo[(all_expo["year"] < int(year)) & (all_expo["crop"] == crop)]
        truth = importer_specific_truth(g, past, q=q, mode=label_mode).reset_index(drop=True)
        base = network_baseline_scores(g, history, trade, int(year), str(crop), lag_years, labels=labels_n3)
        g2 = g.merge(base, on="importer", how="left")
        scores = {
            "forecast": g2["exposure_hat"],
            "oracle": g2["exposure_obs"],
            "n0_lag": g2["n0_lag"] if "n0_lag" in g2.columns else None,
            "n1_mean": g2["n1_mean"] if "n1_mean" in g2.columns else None,
            "n2_hhi": g2["n2_hhi"] if "n2_hhi" in g2.columns else None,
            "n3_propensity": g2["n3_propensity"] if "n3_propensity" in g2.columns else None,
        }
        rec = {
            "year": int(year),
            "crop": crop,
            "lead": int(lead),
            "label_mode": label_mode,
            "n_importers": float(len(g2)),
            "n_high": float(truth.sum()),
            "prevalence": float(truth.mean()) if len(truth) else np.nan,
        }
        for name, score in scores.items():
            if score is None:
                rec[f"auprc_{name}"] = np.nan
                continue
            rec[f"auprc_{name}"] = _auprc(truth, score)
        cont = _continuous_exposure_skill(
            pd.to_numeric(g2["exposure_hat"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(g2["exposure_obs"], errors="coerce").to_numpy(dtype=float),
        )
        rec.update({f"cont_{k}": v for k, v in cont.items()})
        rows.append(rec)
    return pd.DataFrame(rows)


def permute_forecast_auprc(
    pred_panel: pd.DataFrame,
    trade: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    lead: int = 192,
    n_perm: int = 400,
    seed: int = 42,
    lag_years: int = 1,
    q: float = 0.80,
    label_mode: str = "importer_q",
    obs_shocks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Shuffle predicted exporter shocks within year × crop; keep the lagged trade matrix."""

    history = observed_exposure_history(labels, trade, lag_years, obs_shocks=obs_shocks)
    shocks = as_supply_shocks(pred_panel, labels)
    if "shock_hat" not in shocks.columns and "pred" in shocks.columns:
        shocks = shocks.assign(shock_hat=shocks["pred"])
    if obs_shocks is not None:
        shocks = overlay_observed_shocks(shocks, obs_shocks)
    shocks = shocks[shocks["lead"] == int(lead)].copy()
    rng = np.random.default_rng(seed)
    rows = []
    for (year, crop), g in shocks.groupby(["year", "crop"]):
        need = [c for c in ("iso3", "year", "crop", "lead", "shock_hat", "shock_true") if c in g.columns]
        expo = import_exposure(g[need], trade, int(year), str(crop), lag_years)
        if expo.empty:
            continue
        expo = expo.rename(columns={"exposure": "exposure_hat", "flow_true": "exposure_obs"}).reset_index(drop=True)
        past = history[(history["year"] < int(year)) & (history["crop"] == crop)] if len(history) else pd.DataFrame()
        truth = importer_specific_truth(expo, past, q=q, mode=label_mode).reset_index(drop=True)
        real = _auprc(truth, expo["exposure_hat"])
        hats = g["shock_hat"].to_numpy(dtype=float)
        perm_scores = []
        for _ in range(n_perm):
            gg = g.copy()
            gg["shock_hat"] = rng.permutation(hats)
            pexp = import_exposure(gg[need], trade, int(year), str(crop), lag_years)
            if pexp.empty:
                continue
            pexp = pexp.rename(columns={"exposure": "exposure_hat", "flow_true": "exposure_obs"})
            aligned = expo[["importer"]].merge(pexp[["importer", "exposure_hat"]], on="importer", how="left")
            perm_scores.append(_auprc(truth, aligned["exposure_hat"].reset_index(drop=True)))
        arr = np.array(perm_scores, dtype=float)
        rows.append(
            {
                "year": int(year),
                "crop": crop,
                "lead": int(lead),
                "auprc_real": real,
                "auprc_perm_mean": float(np.nanmean(arr)) if len(arr) else np.nan,
                "auprc_perm_p95": float(np.nanquantile(arr, 0.95)) if len(arr) else np.nan,
                "p_ge_real": float(np.nanmean(arr >= real)) if len(arr) and np.isfinite(real) else np.nan,
                "n_perm": int(len(arr)),
            }
        )
    return pd.DataFrame(rows)


def year_block_auprc_ci(
    suite: pd.DataFrame,
    col: str,
    lead: int,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    sub = suite[suite["lead"] == int(lead)]
    if sub.empty or col not in sub.columns:
        return {"p50": np.nan, "p025": np.nan, "p975": np.nan, "p05": np.nan, "p95": np.nan}
    years = [int(y) for y in sorted(sub["year"].unique())]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(years, size=len(years), replace=True)
        vals = []
        for y in pick:
            v = pd.to_numeric(sub.loc[sub["year"] == y, col], errors="coerce")
            if v.notna().any():
                vals.append(float(v.mean()))
        if vals:
            draws.append(float(np.mean(vals)))
    if not draws:
        return {"p50": np.nan, "p025": np.nan, "p975": np.nan, "p05": np.nan, "p95": np.nan}
    s = pd.Series(draws)
    return {
        "p50": float(s.median()),
        "p025": float(s.quantile(0.025)),
        "p975": float(s.quantile(0.975)),
        "p05": float(s.quantile(0.05)),
        "p95": float(s.quantile(0.95)),
    }


def permutation_global(perm: pd.DataFrame) -> dict[str, float]:
    """One p-value on the year–crop mean AUPRC, not the mean of year-wise p-values."""

    if perm is None or perm.empty:
        return {"real": np.nan, "perm_mean": np.nan, "p_ge_real": np.nan}
    real = float(pd.to_numeric(perm["auprc_real"], errors="coerce").mean())
    pmean = float(pd.to_numeric(perm["auprc_perm_mean"], errors="coerce").mean())
    p95 = float(pd.to_numeric(perm["auprc_perm_p95"], errors="coerce").mean()) if "auprc_perm_p95" in perm.columns else np.nan
    p_ge = float(pd.to_numeric(perm["p_ge_real"], errors="coerce").mean())
    return {"real": real, "perm_mean": pmean, "perm_p95_mean": p95, "p_ge_real_mean": p_ge}


def _exposure_from_weights(share: pd.DataFrame, shock: pd.Series) -> pd.Series:
    """Importer exposure = W @ max(0, −shock) for one year–crop matrix."""

    w = share.pivot_table(index="importer", columns="exporter", values="export_share", aggfunc="sum").fillna(0.0)
    s = shock.reindex(w.columns).fillna(0.0).to_numpy(dtype=float)
    val = w.to_numpy(dtype=float) @ np.clip(-s, 0.0, None)
    return pd.Series(val, index=w.index)


def pooled_shock_permutation(
    pred_panel: pd.DataFrame,
    trade: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    lead: int = 192,
    n_perm: int = 2000,
    seed: int = 42,
    lag_years: int = 1,
    q: float = 0.80,
    label_mode: str = "pooled",
    obs_shocks: pd.DataFrame | None = None,
    aggregate: str = "year_crop",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Shuffle predicted shocks within year × crop.

    aggregate='year_crop' averages every year–crop AUPRC.
    aggregate='year_mean' averages crops within a year, then averages years.
    The observed marker uses the same aggregate as the null draws.
    """

    history = observed_exposure_history(labels, trade, lag_years, obs_shocks=obs_shocks)
    shocks = as_supply_shocks(pred_panel, labels)
    if "shock_hat" not in shocks.columns and "pred" in shocks.columns:
        shocks = shocks.assign(shock_hat=shocks["pred"])
    if obs_shocks is not None:
        shocks = overlay_observed_shocks(shocks, obs_shocks)
    shocks = shocks[shocks["lead"] == int(lead)].copy()
    rng = np.random.default_rng(seed)
    prepared = []
    for (year, crop), g in shocks.groupby(["year", "crop"]):
        w = trade[(trade["year"] == int(year) - lag_years) & (trade["crop"] == crop)]
        if w.empty:
            continue
        hat = g.drop_duplicates("iso3").set_index("iso3")["shock_hat"]
        true = g.drop_duplicates("iso3").set_index("iso3")["shock_true"]
        expo_hat = _exposure_from_weights(w, hat)
        expo_obs = _exposure_from_weights(w, true)
        expo = pd.DataFrame({"importer": expo_hat.index, "exposure_hat": expo_hat.to_numpy(), "exposure_obs": expo_obs.reindex(expo_hat.index).to_numpy()})
        past = history[(history["year"] < int(year)) & (history["crop"] == crop)] if len(history) else pd.DataFrame()
        truth = importer_specific_truth(expo, past, q=q, mode=label_mode).reset_index(drop=True)
        share = w[["importer", "exporter", "export_share"]].copy()
        prepared.append((int(year), str(crop), hat.to_numpy(dtype=float), hat.index.to_numpy(), share, expo, truth))

    def _aggregate(pairs: list[tuple[int, float]]) -> float:
        if not pairs:
            return float("nan")
        frame = pd.DataFrame(pairs, columns=["year", "ap"])
        if aggregate == "year_mean":
            return float(frame.groupby("year")["ap"].mean().mean())
        return float(np.nanmean(frame["ap"]))

    real_pairs = [(year, _auprc(truth, expo["exposure_hat"])) for year, _crop, *_rest, expo, truth in prepared]
    real = _aggregate(real_pairs)
    draws = []
    for _ in range(n_perm):
        vals = []
        for year, _crop, hats, exporters, share, expo, truth in prepared:
            perm_hat = pd.Series(rng.permutation(hats), index=exporters)
            aligned = expo[["importer"]].merge(
                _exposure_from_weights(share, perm_hat).rename("exposure_hat").reset_index(),
                on="importer",
                how="left",
            )
            vals.append((year, _auprc(truth, aligned["exposure_hat"].reset_index(drop=True))))
        if vals:
            draws.append(_aggregate(vals))
    arr = np.asarray(draws, dtype=float)
    summary = {
        "observed": real,
        "perm_mean": float(np.nanmean(arr)) if len(arr) else np.nan,
        "perm_p95": float(np.nanquantile(arr, 0.95)) if len(arr) else np.nan,
        "p_ge_obs": empirical_pvalue(arr, real),
        "n_perm": int(len(arr)),
        "n_year_crop": int(len(prepared)),
        "shuffle": "exporters within year × crop",
        "aggregate": aggregate,
    }
    return pd.DataFrame({"auprc_perm": draws}), summary


def exporter_information_ablation(
    pred_panel: pd.DataFrame,
    trade: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    lead: int = 192,
    lag_years: int = 1,
    q: float = 0.80,
    label_mode: str = "pooled",
    n_majors: int = 8,
    obs_shocks: pd.DataFrame | None = None,
    drop_from_label: bool = False,
) -> pd.DataFrame:
    """Zero one major exporter’s predicted shock (optionally also the observed shock)."""

    history = observed_exposure_history(labels, trade, lag_years, obs_shocks=obs_shocks)
    shocks = as_supply_shocks(pred_panel, labels)
    if "shock_hat" not in shocks.columns and "pred" in shocks.columns:
        shocks = shocks.assign(shock_hat=shocks["pred"])
    if obs_shocks is not None:
        shocks = overlay_observed_shocks(shocks, obs_shocks)
    shocks = shocks[shocks["lead"] == int(lead)].copy()
    exp_tot = (
        trade[trade["crop"].isin(tuple(shocks["crop"].unique()))]
        .groupby("exporter")["qty"]
        .sum()
        .sort_values(ascending=False)
    )
    majors = list(exp_tot.head(n_majors).index)

    def _score(frame: pd.DataFrame) -> float:
        vals = []
        for (year, crop), g in frame.groupby(["year", "crop"]):
            need = [c for c in ("iso3", "year", "crop", "lead", "shock_hat", "shock_true") if c in g.columns]
            expo = import_exposure(g[need], trade, int(year), str(crop), lag_years)
            if expo.empty:
                continue
            expo = expo.rename(columns={"exposure": "exposure_hat", "flow_true": "exposure_obs"})
            past = history[(history["year"] < int(year)) & (history["crop"] == crop)]
            truth = importer_specific_truth(expo, past, q=q, mode=label_mode)
            vals.append(_auprc(truth, expo["exposure_hat"]))
        return float(np.nanmean(vals)) if vals else float("nan")

    full = _score(shocks)
    rows = [{"dropped": "none", "auprc": full, "delta": 0.0, "export_qty_share": 1.0, "drop_from_label": drop_from_label}]
    total_qty = float(exp_tot.sum()) if len(exp_tot) else np.nan
    for j in majors:
        gg = shocks.copy()
        gg.loc[gg["iso3"] == j, "shock_hat"] = 0.0
        if drop_from_label:
            gg.loc[gg["iso3"] == j, "shock_true"] = 0.0
        a = _score(gg)
        rows.append(
            {
                "dropped": j,
                "auprc": a,
                "delta": full - a,
                "export_qty_share": float(exp_tot.loc[j] / total_qty) if j in exp_tot.index else np.nan,
                "drop_from_label": drop_from_label,
            }
        )
    return pd.DataFrame(rows)


def _assoc(y: np.ndarray, x: np.ndarray) -> dict[str, float]:
    from cropwarn.validation.metrics import regression_metrics
    from scipy.stats import pearsonr, spearmanr

    rec = regression_metrics(y, x)
    m = np.isfinite(y) & np.isfinite(x)
    rec["pearson"] = float("nan")
    rec["spearman"] = float("nan")
    if m.sum() >= 8:
        try:
            rec["pearson"] = float(pearsonr(y[m], x[m]).statistic)
        except Exception:
            pass
        try:
            rec["spearman"] = float(spearmanr(y[m], x[m]).statistic)
        except Exception:
            pass
    return rec


def yield_persistence(labels: pd.DataFrame, crops: tuple[str, ...] = ("wheat", "soybean")) -> dict[str, float]:
    lab = labels[labels["crop"].isin(crops)].copy()
    if "yield_anomaly" not in lab.columns:
        return {}
    keys = ["iso3", "crop", "year"]
    cur = lab[keys + ["yield_anomaly"]].drop_duplicates()
    prev = cur.copy()
    prev["year"] = prev["year"] + 1
    prev = prev.rename(columns={"yield_anomaly": "y_lag1"})
    m = cur.merge(prev, on=keys, how="inner")
    m = m[m["year"] >= 2016]
    out = {"pooled": _assoc(m["yield_anomaly"].to_numpy(), m["y_lag1"].to_numpy())}
    for crop, g in m.groupby("crop"):
        out[str(crop)] = _assoc(g["yield_anomaly"].to_numpy(), g["y_lag1"].to_numpy())
    return out


def lagged_optical_skill(
    series: pd.DataFrame,
    labels: pd.DataFrame,
    var: str = "evi",
    crops: tuple[str, ...] = ("wheat", "soybean"),
) -> dict:
    """P1: previous-year optical seasonal mean, fitted on years < t (rolling origin)."""

    if var not in series.columns:
        return {}
    seas = series.groupby(["iso3", "crop", "year"], as_index=False)[var].mean()
    lab = labels[labels["crop"].isin(crops)][["iso3", "crop", "year", "yield_anomaly"]].drop_duplicates()
    rows = []
    for year in sorted(y for y in lab["year"].unique() if int(y) >= 2016):
        train = lab[lab["year"] < int(year)].merge(
            seas.rename(columns={var: "x", "year": "xyear"}).assign(year=lambda d: d["xyear"] + 1)[
                ["iso3", "crop", "year", "x"]
            ],
            on=["iso3", "crop", "year"],
            how="inner",
        )
        test = lab[lab["year"] == int(year)].merge(
            seas.rename(columns={var: "x", "year": "xyear"}).assign(year=lambda d: d["xyear"] + 1)[
                ["iso3", "crop", "year", "x"]
            ],
            on=["iso3", "crop", "year"],
            how="inner",
        )
        train = train.dropna(subset=["yield_anomaly", "x"])
        test = test.dropna(subset=["yield_anomaly", "x"])
        if len(train) < 12 or test.empty:
            continue
        coef = np.polyfit(train["x"].to_numpy(dtype=float), train["yield_anomaly"].to_numpy(dtype=float), 1)
        pred = coef[0] * test["x"].to_numpy(dtype=float) + coef[1]
        rec = _assoc(test["yield_anomaly"].to_numpy(dtype=float), pred)
        rec["year"] = int(year)
        rec["n_train"] = float(len(train))
        rows.append(rec)
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    return {
        "by_year": df.to_dict("records"),
        "mean_r2": float(df["r2"].mean()),
        "mean_pearson": float(df["pearson"].mean()) if "pearson" in df.columns else np.nan,
        "n_years": float(len(df)),
    }
