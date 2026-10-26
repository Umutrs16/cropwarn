"""Baseline, Chen-style clustered RF, XGBoost and phenology-aware temporal encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cropwarn.constants import MAIZE_SOUTH_ISO3
from cropwarn.features.factory import feature_columns


def _xy(df: pd.DataFrame, target: str | None = None, use_country_id: bool = False) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    cols = feature_columns(df, use_country_id=use_country_id)
    x = df[cols].copy()
    all_nan = [c for c in x.columns if x[c].isna().all()]
    if all_nan:
        x = x.drop(columns=all_nan)
    sparse = [c for c in x.columns if x[c].isna().mean() > 0.85]
    if sparse:
        x = x.drop(columns=sparse)
    cat_cols = [c for c in ("iso3", "crop", "region", "latitude_band") if c in x.columns]
    if cat_cols:
        x = pd.get_dummies(x, columns=cat_cols, dummy_na=False)
    y = df[target].to_numpy(dtype=float) if target and target in df.columns else np.array([])
    return x, y, list(x.columns)


def _align_columns(x: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return x.reindex(columns=columns, fill_value=0.0)


@dataclass
class FittedModel:
    name: str
    estimator: Any
    columns: list[str]
    extra: dict


class ClimatologyModel:
    def fit(self, train: pd.DataFrame, target: str) -> FittedModel:
        keys = ["unit_id", "crop"] if "unit_id" in train.columns else ["iso3", "crop"]
        means = train.groupby(keys)[target].mean().to_dict()
        global_mean = float(np.nanmean(train[target]))
        return FittedModel("climatology", (means, global_mean), [], {"keys": keys})

    def predict(self, fitted: FittedModel, test: pd.DataFrame) -> np.ndarray:
        means, global_mean = fitted.estimator
        keys = fitted.extra.get("keys", ["iso3", "crop"])

        def _key(r):
            uid = getattr(r, "unit_id", None) or r.iso3
            return (uid, r.crop) if keys[0] == "unit_id" else (r.iso3, r.crop)

        return np.array([means.get(_key(r), global_mean) for r in test.itertuples(index=False)])


class TrendModel:
    def fit(self, train: pd.DataFrame, target: str) -> FittedModel:
        models = {}
        keys = ["unit_id", "crop"] if "unit_id" in train.columns else ["iso3", "crop"]
        for key, g in train.groupby(keys):
            if g["year"].nunique() < 3:
                continue
            models[key] = LinearRegression().fit(g[["year"]], g[target])
        return FittedModel("trend", models, [], {"keys": keys})

    def predict(self, fitted: FittedModel, test: pd.DataFrame) -> np.ndarray:
        keys = fitted.extra.get("keys", ["iso3", "crop"])
        out = []
        for r in test.itertuples(index=False):
            key = (getattr(r, "unit_id", None) or r.iso3, r.crop) if keys[0] == "unit_id" else (r.iso3, r.crop)
            m = fitted.estimator.get(key)
            out.append(float(m.predict([[r.year]])[0]) if m is not None else 0.0)
        return np.array(out)


class ChenClusterRF:
    """Phenology-clustered random forests, following the Chen et al. 2026 idea."""

    def __init__(self, n_estimators: int = 200, n_clusters: int = 6, seed: int = 42):
        self.n_estimators = n_estimators
        self.n_clusters = n_clusters
        self.seed = seed

    def fit(self, train: pd.DataFrame, target: str) -> FittedModel:
        keys = ["unit_id", "crop"] if "unit_id" in train.columns else ["iso3", "crop"]
        proto = (
            train.groupby(keys, as_index=False)[["sos_static", "eos_static"]]
            .median()
        )
        n_c = min(self.n_clusters, len(proto))
        if n_c < 2:
            proto["cluster"] = 0
            km = None
        else:
            km = KMeans(n_clusters=n_c, random_state=self.seed, n_init=10)
            proto["cluster"] = km.fit_predict(proto[["sos_static", "eos_static"]])
        merged = train.merge(proto[[*keys, "cluster"]], on=keys, how="left")
        x, y, cols = _xy(merged, target)
        forests = {}
        for cid, g in merged.groupby("cluster"):
            xg, yg, _ = _xy(g, target)
            xg = _align_columns(xg, cols)
            pipe = Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    (
                        "rf",
                        RandomForestRegressor(
                            n_estimators=self.n_estimators,
                            min_samples_leaf=5,
                            max_features="sqrt",
                            random_state=self.seed,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            mask = np.isfinite(yg)
            if mask.sum() < 20:
                continue
            pipe.fit(xg.loc[mask], yg[mask])
            forests[int(cid)] = pipe
        fallback = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        random_state=self.seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        mask = np.isfinite(y)
        fallback.fit(x.loc[mask], y[mask])
        return FittedModel("chen_rf", (km, proto, forests, fallback), cols, {"keys": keys})

    def predict(self, fitted: FittedModel, test: pd.DataFrame) -> np.ndarray:
        km, proto, forests, fallback = fitted.estimator
        keys = fitted.extra.get("keys", ["iso3", "crop"])
        tes = test.merge(proto[[*keys, "cluster"]], on=keys, how="left")
        missing = tes["cluster"].isna()
        if missing.any() and km is not None:
            tes.loc[missing, "cluster"] = km.predict(
                tes.loc[missing, ["sos_static", "eos_static"]].fillna(180)
            )
        tes["cluster"] = tes["cluster"].fillna(0)
        pred = np.zeros(len(tes))
        for cid, g in tes.groupby("cluster"):
            model = forests.get(int(cid), fallback)
            yp = model.predict(_align_columns(_xy(g)[0], fitted.columns))
            pred[tes.index.get_indexer(g.index)] = yp
        return pred


class XGBModel:
    def __init__(self, n_estimators: int = 200, seed: int = 42):
        self.n_estimators = n_estimators
        self.seed = seed

    def fit(self, train: pd.DataFrame, target: str) -> FittedModel:
        from xgboost import XGBRegressor

        x, y, cols = _xy(train, target)
        mask = np.isfinite(y)
        model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.seed,
            n_jobs=-1,
            tree_method="hist",
        )
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("xgb", model)])
        pipe.fit(x.loc[mask], y[mask])
        return FittedModel("xgb", pipe, cols, {})

    def predict(self, fitted: FittedModel, test: pd.DataFrame) -> np.ndarray:
        x, _, _ = _xy(test)
        x = _align_columns(x, fitted.columns)
        return fitted.estimator.predict(x)


class TemporalEncoder:
    """Phenology-aligned multimodal encoder.

    Stage windows are treated as a short sequence. A residual ExtraTrees /
    MLP stack captures temporal interactions without requiring a GPU.
    """

    def __init__(self, seed: int = 42, hidden: int = 64, n_estimators: int = 250):
        self.seed = seed
        self.hidden = hidden
        self.n_estimators = n_estimators

    def fit(self, train: pd.DataFrame, target: str) -> FittedModel:
        x, y, cols = _xy(train, target)
        mask = np.isfinite(y)
        tree = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                (
                    "et",
                    ExtraTreesRegressor(
                        n_estimators=self.n_estimators,
                        min_samples_leaf=3,
                        random_state=self.seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        mlp = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(self.hidden, self.hidden // 2),
                        activation="relu",
                        max_iter=220,
                        random_state=self.seed,
                        early_stopping=True,
                    ),
                ),
            ]
        )
        tree.fit(x.loc[mask], y[mask])
        # MLP is optional and often unstable on wide, small agricultural panels.
        use_mlp = False
        try:
            if mask.sum() > 400:
                mlp.fit(x.loc[mask], y[mask])
                use_mlp = True
            else:
                mlp = None
        except Exception:
            mlp = None
        return FittedModel("temporal", (tree, mlp), cols, {"use_mlp": use_mlp})

    def predict(self, fitted: FittedModel, test: pd.DataFrame) -> np.ndarray:
        x, _, _ = _xy(test)
        x = _align_columns(x, fitted.columns)
        tree, mlp = fitted.estimator
        p = tree.predict(x)
        if mlp is not None:
            p = 0.65 * p + 0.35 * mlp.predict(x)
        return p


MODEL_REGISTRY = {
    "climatology": ClimatologyModel,
    "trend": TrendModel,
    "chen_rf": ChenClusterRF,
    "xgb": XGBModel,
    "temporal": TemporalEncoder,
}


def make_model(name: str, demo: bool, cfg) -> Any:
    if name == "chen_rf":
        n = cfg.get("models", "rf", "demo_n_estimators" if demo else "n_estimators")
        return ChenClusterRF(n_estimators=int(n), n_clusters=int(cfg.get("clusters", "n_clusters")), seed=cfg.seed)
    if name == "xgb":
        n = cfg.get("models", "xgb", "demo_n_estimators" if demo else "n_estimators")
        return XGBModel(n_estimators=int(n), seed=cfg.seed)
    if name == "temporal":
        return TemporalEncoder(
            seed=cfg.seed,
            hidden=int(cfg.get("models", "temporal", "hidden")),
            n_estimators=80 if demo else 250,
        )
    if name == "climatology":
        return ClimatologyModel()
    if name == "trend":
        return TrendModel()
    raise KeyError(name)


def _fit_predict_one(name: str, train: pd.DataFrame, test: pd.DataFrame, target: str, demo: bool, cfg) -> np.ndarray:
    model = make_model(name, demo, cfg)
    fitted = model.fit(train, target)
    return model.predict(fitted, test)


def fit_predict(name: str, train: pd.DataFrame, test: pd.DataFrame, target: str, demo: bool, cfg) -> np.ndarray:
    """Fit one model per crop. Pooled training mixes flooded rice with dryland wheat."""

    if "crop" not in train.columns or train["crop"].nunique() <= 1:
        return _fit_predict_one(name, train, test, target, demo, cfg)
    test = test.reset_index(drop=True)
    out = np.full(len(test), np.nan, dtype=float)

    def _group_key(frame: pd.DataFrame) -> pd.Series:
        key = frame["crop"].astype(str)
        maize_s = (frame["crop"] == "maize") & frame["iso3"].isin(MAIZE_SOUTH_ISO3)
        key = key.where(~maize_s, "maize_south")
        return key

    test = test.copy()
    train = train.copy()
    test["_mk"] = _group_key(test)
    train["_mk"] = _group_key(train)
    for key, te in test.groupby("_mk", sort=False):
        tr = train[train["_mk"] == key]
        if te.empty:
            continue
        if len(tr) < 20:
            tr = train
        pred_c = _fit_predict_one(
            name,
            tr.drop(columns=["_mk"]).reset_index(drop=True),
            te.drop(columns=["_mk"]).reset_index(drop=True),
            target,
            demo,
            cfg,
        )
        out[te.index.to_numpy()] = pred_c
    if not np.isfinite(out).all():
        fallback = _fit_predict_one(name, train, test, target, demo, cfg)
        out = np.where(np.isfinite(out), out, fallback)
    return out
