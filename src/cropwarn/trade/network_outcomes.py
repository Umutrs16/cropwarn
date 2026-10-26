"""Importer-network consequences from the lagged trade matrix (no new FAOSTAT domains)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cropwarn.trade.exposure import evaluate_realized_import_shortfall, high_risk_detection, import_exposure


def importer_structure(trade: pd.DataFrame) -> pd.DataFrame:
    if trade is None or trade.empty or "qty" not in trade.columns:
        return pd.DataFrame()
    g = trade.copy()
    tot = g.groupby(["year", "crop", "importer"])["qty"].transform("sum")
    g["share"] = g["qty"] / tot.replace(0, np.nan)
    rows = []
    for (year, crop, importer), sub in g.groupby(["year", "crop", "importer"]):
        shares = pd.to_numeric(sub["share"], errors="coerce").dropna()
        if shares.empty:
            continue
        rows.append(
            {
                "year": int(year),
                "crop": crop,
                "importer": importer,
                "hhi": float((shares**2).sum()),
                "top1_share": float(shares.max()),
                "n_suppliers": int((shares > 0).sum()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["importer", "crop", "year"])
    out["hhi_chg"] = out.groupby(["importer", "crop"])["hhi"].diff()
    out["top1_chg"] = out.groupby(["importer", "crop"])["top1_share"].diff()
    out["n_suppliers_chg"] = out.groupby(["importer", "crop"])["n_suppliers"].diff()
    return out


def evaluate_structure_outcomes(
    pred_panel: pd.DataFrame,
    trade: pd.DataFrame,
    lag_years: int = 1,
    high_q: float = 0.80,
) -> pd.DataFrame:
    """Does predicted exposure rank realized HHI rise / top-supplier change / supplier loss?"""

    struct = importer_structure(trade)
    if struct.empty:
        return pd.DataFrame()
    rows = []
    for (year, crop, lead), g in pred_panel.groupby(["year", "crop", "lead"]):
        shocks = g.rename(columns={"pred": "shock_hat"})
        keep = [c for c in ("iso3", "year", "crop", "lead", "shock_hat", "shock_true") if c in shocks.columns]
        expo = import_exposure(shocks[keep], trade, int(year), str(crop), lag_years)
        if expo.empty:
            continue
        obs = struct[(struct["year"] == int(year)) & (struct["crop"] == crop)]
        expo = expo.merge(obs, on="importer", how="left")
        s = pd.to_numeric(expo["exposure"], errors="coerce")
        for name, col, high_is_bad in (
            ("hhi_increase", "hhi_chg", True),
            ("top1_increase", "top1_chg", True),
            ("supplier_loss", "n_suppliers_chg", False),
        ):
            y = pd.to_numeric(expo[col], errors="coerce")
            m = y.notna() & s.notna()
            if m.sum() < 8:
                continue
            yy = y[m]
            truth = yy >= yy.quantile(0.80) if high_is_bad else yy <= yy.quantile(0.20)
            scored = expo.loc[m].assign(flow_true=yy if high_is_bad else -yy)
            det = high_risk_detection(scored, high_q)
            pred_flag = s[m] >= s[m].quantile(high_q)
            tp = float((truth.to_numpy() & pred_flag.to_numpy()).sum())
            fn = float((truth.to_numpy() & ~pred_flag.to_numpy()).sum())
            rec = tp / max(tp + fn, 1.0)
            from sklearn.metrics import average_precision_score

            try:
                auprc = float(average_precision_score(truth.astype(int), s[m]))
            except ValueError:
                auprc = float("nan")
            rows.append(
                {
                    "year": int(year),
                    "crop": crop,
                    "lead": int(lead),
                    "outcome": name,
                    "recall": rec,
                    "auprc": auprc,
                    "precision": det.get("precision", np.nan),
                    "n": float(m.sum()),
                }
            )
    return pd.DataFrame(rows)
