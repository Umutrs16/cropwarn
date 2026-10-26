"""Stage 12 robustness: alternate labels, country-ID sensitivity, rice without VOD."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cropwarn.config import Config
from cropwarn.data.io import write_table
from cropwarn.data.labels import build_labels
from cropwarn.features.factory import feature_columns
from cropwarn.validation.leakage import assert_no_label_columns


def drop_country_identity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("iso3", "country", "faostat_code", "tech_level"):
        if col in out.columns:
            out = out.drop(columns=[col])
    return out


def alternate_detrend(panel: pd.DataFrame, method: str = "quadratic") -> pd.DataFrame:
    drop = [c for c in panel.columns if c.startswith("yield_") or c.startswith("production_")]
    keep = panel.drop(columns=drop, errors="ignore")
    if "yield_tha" not in keep.columns and "yield_tha" in panel.columns:
        keep = keep.assign(yield_tha=panel["yield_tha"])
    return build_labels(keep, detrend_method=method)


def exclude_rice_vod(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    vod_cols = [c for c in out.columns if c.startswith("vod_")]
    out.loc[out["crop"] == "rice", vod_cols] = float("nan")
    return out


def audit_feature_contract(df: pd.DataFrame) -> list[str]:
    cols = feature_columns(df)
    assert_no_label_columns(cols)
    return cols


def _lead_summary(lead_skill: pd.DataFrame, label: str) -> pd.DataFrame:
    keep = lead_skill[lead_skill["experiment"].isin(["B2_optical_climate", "B4_plus_sif", "P_food_system"])].copy()
    harvest = keep[keep["lead"] == 0][["experiment", "r2"]].rename(columns={"r2": "harvest_r2"})
    leads = keep.groupby("experiment", as_index=False)["earliest_lead"].first()
    out = harvest.merge(leads, on="experiment")
    out["variant"] = label
    return out


def run_robustness_suite(
    world: dict,
    cfg: Config,
    out_dir: str | Path,
    methods: list[str] | None = None,
) -> dict:
    """B2 / B4 / food-system only, rolling-origin. Does not overwrite official results/."""

    from cropwarn.experiments.runner import run_experiment_suite

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    experiments = ["B2_optical_climate", "B4_plus_sif", "P_food_system"]
    rows = []
    methods = methods or ["quadratic"]

    official_note = (
        "Official features already drop country identity "
        "(feature_columns use_country_id=False). That check is the default, not a new run."
    )
    (out / "README.md").write_text(
        "# Stage 12 robustness\n\n"
        + official_note
        + "\n\nLabel variants requested: "
        + ", ".join(methods)
        + ". Quadratic cache lives under alternate_detrend/. "
        "TROPOMI/GOME-2 independence is not available (inventory.tropomi_sif=false). "
        "Rice-without-VOD is N/A on the wheat+soy headline.\n",
        encoding="utf-8",
    )

    bundles: dict = {}
    for method in methods:
        dest = out / ("alternate_detrend" if method == "quadratic" else f"label_{method}")
        variant_world = dict(world)
        variant_world["panel"] = alternate_detrend(world["panel"], method=method)
        bundle = run_experiment_suite(
            variant_world,
            cfg,
            demo=True,
            schemes=["rolling_origin"],
            results_dir=dest,
            experiments=experiments,
        )
        rows.append(_lead_summary(bundle["lead_skill"], f"{method}_detrend"))
        bundles[method] = bundle
    summary = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(summary):
        write_table(summary, out / "robustness_summary.csv")
    bundles["summary"] = summary
    return bundles
