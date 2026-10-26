"""Point, anomaly-warning, uncertainty and trade metrics from section 6.2."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score


def _finite(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(y) & np.isfinite(p)
    return y[m], p[m]


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y, pred = _finite(np.asarray(y, float), np.asarray(pred, float))
    if len(y) < 3:
        return {k: np.nan for k in ("n", "r2", "rmse", "mae", "nrmse", "mape")}
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    mae = float(mean_absolute_error(y, pred))
    denom = float(np.mean(np.abs(y))) + 1e-6
    mape = float(np.mean(np.abs((y - pred) / (np.abs(y) + 1e-6))))
    return {
        "n": float(len(y)),
        "r2": float(r2_score(y, pred)),
        "rmse": rmse,
        "mae": mae,
        "nrmse": rmse / denom,
        "mape": mape,
    }


def classification_metrics(y_true_sign: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true_sign)
    s = np.asarray(scores)
    m = np.isfinite(y) & np.isfinite(s)
    y, s = y[m], s[m]
    if len(y) < 8 or len(np.unique(y)) < 2:
        return {k: np.nan for k in ("sign_acc", "auroc", "auprc", "top_decile_recall")}
    pred_sign = (s < 0).astype(int)
    true_loss = (y < 0).astype(int)
    sign_acc = float(np.mean((np.sign(s) == np.sign(y)) | ((s == 0) & (y == 0))))
    try:
        auroc = float(roc_auc_score(true_loss, -s))
    except ValueError:
        auroc = np.nan
    try:
        auprc = float(average_precision_score(true_loss, -s))
    except ValueError:
        auprc = np.nan
    k = max(1, int(0.10 * len(s)))
    idx = np.argsort(s)[:k]
    top_recall = float(true_loss[idx].sum() / max(true_loss.sum(), 1))
    return {
        "sign_acc": sign_acc,
        "auroc": auroc,
        "auprc": auprc,
        "top_decile_recall": top_recall,
        "loss_n": float(true_loss.sum()),
    }


def earliest_reliable_lead(
    lead_metrics: pd.DataFrame,
    r2_floor: float = 0.15,
    max_width: float | None = None,
) -> float:
    """Longest isolated lead with R² ≥ floor. Prefer stable_reliable_lead for operations."""

    ok = lead_metrics[(lead_metrics["r2"] >= r2_floor) & (lead_metrics["r2"] < 1.5)]
    if max_width is not None and "mean_width" in lead_metrics:
        ok = ok[ok["mean_width"] <= max_width]
    if ok.empty:
        return np.nan
    return float(ok["lead"].max())


def stable_reliable_lead(
    lead_metrics: pd.DataFrame,
    r2_floor: float = 0.15,
    min_consecutive: int = 1,
) -> float:
    """Longest L such that every lead ≤ L has R² ≥ floor (contiguous to harvest).

    If min_consecutive > 1, require that many successive windows at the outer edge.
    Isolated long-lead spikes do not count.
    """

    if lead_metrics is None or lead_metrics.empty or "lead" not in lead_metrics:
        return float("nan")
    g = lead_metrics.sort_values("lead")
    ok = (pd.to_numeric(g["r2"], errors="coerce") >= r2_floor) & (pd.to_numeric(g["r2"], errors="coerce") < 1.5)
    leads = pd.to_numeric(g["lead"], errors="coerce").to_numpy()
    flags = ok.to_numpy()
    stable = float("nan")
    run = 0
    for lead, good in zip(leads, flags):
        if good:
            run += 1
            if run >= min_consecutive:
                stable = float(lead)
            continue
        break
    return stable


def robust_stable_lead(
    lead_metrics: pd.DataFrame,
    r2_floor: float = 0.15,
    bound_col: str = "r2_p05",
) -> float:
    """Longest L such that every lead ≤ L has a lower R² bound ≥ floor.

    Use bootstrap 5th percentiles, not point estimates. Isolated long-lead
    spikes and 0.006-threshold hops do not count.
    """

    if lead_metrics is None or lead_metrics.empty or "lead" not in lead_metrics:
        return float("nan")
    col = bound_col if bound_col in lead_metrics.columns else "r2"
    if col not in lead_metrics.columns:
        return float("nan")
    g = lead_metrics.sort_values("lead")
    ok = (pd.to_numeric(g[col], errors="coerce") >= r2_floor) & (pd.to_numeric(g[col], errors="coerce") < 1.5)
    leads = pd.to_numeric(g["lead"], errors="coerce").to_numpy()
    flags = ok.to_numpy()
    stable = float("nan")
    for lead, good in zip(leads, flags):
        if good:
            stable = float(lead)
            continue
        break
    return stable


def skill_score(mse_model: float, mse_ref: float) -> float:
    if not np.isfinite(mse_model) or not np.isfinite(mse_ref) or mse_ref <= 0:
        return float("nan")
    return float(1.0 - mse_model / mse_ref)


def selective_risk(abs_err: np.ndarray, ood: np.ndarray, coverages: tuple[float, ...] = (0.5, 0.7, 0.9)) -> dict[str, float]:
    order = np.argsort(ood)
    err = abs_err[order]
    out = {}
    for c in coverages:
        n = max(1, int(len(err) * c))
        out[f"risk_at_{int(c * 100)}"] = float(np.mean(err[:n]))
    return out
