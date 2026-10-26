"""Import-dependency and food-system buffer diagnostics.

The trade-matrix column ``import_dependency`` is a relative-volume placeholder
and is not used here. IDR is reconstructed from FAOSTAT production and
bilateral quantities:

    IDR = Imports / (Production + Imports − Exports)

using year t−1 for any forecast year t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def importer_accounts(trade: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Country–crop–year production, imports, exports and IDR."""

    t = trade.copy()
    t["qty"] = pd.to_numeric(t["qty"], errors="coerce")
    imports = (
        t.groupby(["year", "crop", "importer"], as_index=False)["qty"]
        .sum()
        .rename(columns={"importer": "iso3", "qty": "imports"})
    )
    exports = (
        t.groupby(["year", "crop", "exporter"], as_index=False)["qty"]
        .sum()
        .rename(columns={"exporter": "iso3", "qty": "exports"})
    )
    lab = labels.copy()
    prod_col = "production_t" if "production_t" in lab.columns else "production"
    prod = lab.groupby(["year", "crop", "iso3"], as_index=False)[prod_col].first()
    prod = prod.rename(columns={prod_col: "production"})
    out = imports.merge(exports, on=["year", "crop", "iso3"], how="outer")
    out = out.merge(prod, on=["year", "crop", "iso3"], how="left")
    for col in ("imports", "exports", "production"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    supply = out["production"] + out["imports"] - out["exports"]
    idr = np.where(supply > 0, out["imports"] / supply, 0.0)
    out["idr"] = np.clip(idr, 0.0, 1.0)
    out["self_sufficiency"] = np.where(
        (out["production"] + out["imports"]) > 0,
        out["production"] / (out["production"] + out["imports"]),
        np.nan,
    )
    return out


def lag_accounts(accounts: pd.DataFrame) -> pd.DataFrame:
    """Attach year t−1 accounts to year t (no contemporaneous leak)."""

    lag = accounts.copy()
    lag["year"] = lag["year"] + 1
    return lag.rename(
        columns={
            "imports": "imports_lag1",
            "exports": "exports_lag1",
            "production": "production_lag1",
            "idr": "idr_lag1",
            "self_sufficiency": "self_sufficiency_lag1",
        }
    )


def importer_hhi_lag(trade: pd.DataFrame) -> pd.DataFrame:
    g = trade.copy()
    tot = g.groupby(["year", "crop", "importer"])["qty"].transform("sum")
    g["share"] = pd.to_numeric(g["qty"], errors="coerce") / tot.replace(0, np.nan)
    rows = (
        g.groupby(["year", "crop", "importer"], as_index=False)
        .agg(
            hhi=("share", lambda s: float(np.nansum(np.square(pd.to_numeric(s, errors="coerce"))))),
            n_suppliers=("share", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
        )
    )
    rows["year"] = rows["year"] + 1
    return rows.rename(
        columns={
            "importer": "iso3",
            "hhi": "hhi_lag1",
            "n_suppliers": "n_suppliers_lag1",
        }
    )


def attach_buffers(expo: pd.DataFrame, accounts_lag: pd.DataFrame, hhi_lag: pd.DataFrame) -> pd.DataFrame:
    out = expo.copy()
    key_imp = "importer" if "importer" in out.columns else "iso3"
    acc = accounts_lag.rename(columns={"iso3": key_imp})
    hh = hhi_lag.rename(columns={"iso3": key_imp})
    out = out.merge(acc, on=["year", "crop", key_imp], how="left")
    out = out.merge(hh, on=["year", "crop", key_imp], how="left")
    hat = pd.to_numeric(out.get("exposure_hat", out.get("exposure")), errors="coerce")
    obs = pd.to_numeric(out.get("exposure_obs", out.get("flow_true")), errors="coerce")
    idr = pd.to_numeric(out.get("idr_lag1"), errors="coerce")
    out["supply_hat"] = hat * idr
    out["supply_obs"] = obs * idr
    return out


def _auprc(truth: pd.Series, score: pd.Series) -> float:
    y = truth.astype(int)
    s = pd.to_numeric(score, errors="coerce")
    m = s.notna() & y.notna()
    if m.sum() < 8 or y[m].nunique() < 2:
        return float("nan")
    try:
        return float(average_precision_score(y[m], s[m]))
    except ValueError:
        return float("nan")


def median_split_auprc(
    df: pd.DataFrame,
    split_col: str,
    truth: pd.Series,
    score: pd.Series,
) -> pd.DataFrame:
    x = pd.to_numeric(df[split_col], errors="coerce")
    mid = float(x.median()) if x.notna().any() else np.nan
    rows = []
    for name, mask in (("low", x <= mid), ("high", x > mid), ("all", x.notna())):
        rows.append(
            {
                "split": split_col,
                "stratum": name,
                "threshold": mid,
                "n": float(mask.sum()),
                "prevalence": float(truth[mask].mean()) if mask.any() else np.nan,
                "auprc": _auprc(truth[mask], score[mask]),
            }
        )
    return pd.DataFrame(rows)


def shortfall_given_exposure(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    shortfall: pd.Series,
    high_expo: pd.Series,
    n_bin: int = 2,
) -> pd.DataFrame:
    """P(shortfall | high predicted exposure) on a row_col × col_col grid."""

    r = pd.to_numeric(df[row_col], errors="coerce")
    c = pd.to_numeric(df[col_col], errors="coerce")
    rq = pd.qcut(r.rank(method="first"), n_bin, labels=["low", "high"], duplicates="drop")
    cq = pd.qcut(c.rank(method="first"), n_bin, labels=["low", "high"], duplicates="drop")
    rows = []
    for rv in ("low", "high"):
        for cv in ("low", "high"):
            m = (rq == rv) & (cq == cv) & high_expo.notna() & shortfall.notna()
            he = high_expo[m]
            if he.sum() < 5:
                p = np.nan
            else:
                p = float(shortfall[m & he].mean())
            rows.append(
                {
                    "row": rv,
                    "col": cv,
                    "n": float(m.sum()),
                    "n_high_expo": float(he.sum()) if m.any() else 0.0,
                    "p_shortfall_given_high_expo": p,
                    "p_shortfall": float(shortfall[m].mean()) if m.any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def skill_score(auprc: float, prevalence: float) -> float:
    if not np.isfinite(auprc) or not np.isfinite(prevalence) or prevalence >= 1:
        return float("nan")
    return float((auprc - prevalence) / (1.0 - prevalence))
