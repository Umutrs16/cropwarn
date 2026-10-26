"""Quantile residuals, split conformal intervals and OOD screens."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from cropwarn.features.factory import feature_columns


@dataclass
class ConformalResult:
    pred: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    qhat: float
    width: np.ndarray


def split_conformal(
    y_cal: np.ndarray,
    pred_cal: np.ndarray,
    pred_test: np.ndarray,
    alpha: float = 0.10,
) -> ConformalResult:
    resid = np.abs(y_cal - pred_cal)
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        qhat = 0.0
    else:
        qhat = float(np.quantile(resid, np.clip((1 - alpha) * (1 + 1 / resid.size), 0, 1)))
    lower = pred_test - qhat
    upper = pred_test + qhat
    return ConformalResult(pred_test, lower, upper, qhat, upper - lower)


def empirical_coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(lower) & np.isfinite(upper)
    if m.sum() == 0:
        return np.nan
    return float(np.mean((y[m] >= lower[m]) & (y[m] <= upper[m])))


def crps_gaussian_proxy(y: np.ndarray, pred: np.ndarray, width: np.ndarray) -> float:
    """CRPS-like score using interval width as a scale proxy."""

    sigma = np.clip(width / 3.2897, 1e-4, None)  # ~90% normal interval
    z = (y - pred) / sigma
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    from scipy.special import erf

    cdf = 0.5 * (1.0 + erf(z / np.sqrt(2)))
    crps = sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
    crps = crps[np.isfinite(crps)]
    return float(np.mean(crps)) if crps.size else np.nan


class OODDetector:
    def __init__(self, n_components: int = 12, n_neighbors: int = 15):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.imp = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=0)
        self.cov = EmpiricalCovariance()
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.energy_mu: np.ndarray | None = None
        self.keep_cols: list[str] = []

    def _x(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        cols = [c for c in feature_columns(df) if c not in {"crop", "region", "latitude_band", "iso3"}]
        x = df[cols].select_dtypes(include=[np.number])
        if fit:
            self.keep_cols = [c for c in x.columns if not x[c].isna().all()]
        use = self.keep_cols or [c for c in x.columns if not x[c].isna().all()]
        return x.reindex(columns=use, fill_value=np.nan).to_numpy(dtype=float)

    def fit(self, train: pd.DataFrame) -> "OODDetector":
        x = self.imp.fit_transform(self._x(train, fit=True))
        x = self.scaler.fit_transform(x)
        ncomp = min(self.n_components, x.shape[1], max(2, x.shape[0] - 1))
        self.pca = PCA(n_components=ncomp, random_state=0)
        z = self.pca.fit_transform(x)
        self.cov.fit(z)
        k = min(self.n_neighbors, max(1, len(z) - 1))
        self.nn = NearestNeighbors(n_neighbors=k)
        self.nn.fit(z)
        self.energy_mu = z.mean(axis=0)
        return self

    def scores(self, test: pd.DataFrame) -> pd.DataFrame:
        x = self.scaler.transform(self.imp.transform(self._x(test)))
        z = self.pca.transform(x)
        maha = self.cov.mahalanobis(z)
        dist, _ = self.nn.kneighbors(z)
        knn = dist.mean(axis=1)
        energy = np.linalg.norm(z - self.energy_mu, axis=1)
        return pd.DataFrame(
            {
                "ood_mahalanobis": maha,
                "ood_knn": knn,
                "ood_energy": energy,
                "ood_score": 0.5 * _rank01(maha) + 0.3 * _rank01(knn) + 0.2 * _rank01(energy),
            }
        )


def _rank01(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x))
    return order / max(len(x) - 1, 1)


def assign_trust(ood_score: np.ndarray, reject_q: float = 0.90) -> np.ndarray:
    hi = np.quantile(ood_score, reject_q) if len(ood_score) else 1.0
    mid = np.quantile(ood_score, 0.70) if len(ood_score) else 0.7
    out = np.full(len(ood_score), "reliable", dtype=object)
    out[ood_score >= mid] = "high_risk"
    out[ood_score >= hi] = "reject"
    return out
