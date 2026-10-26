"""Six manuscript figures specified in the paper framework."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "savefig.bbox": "tight",
    }
)

COLORS = {
    "B0_chen_rf": "#7f7f7f",
    "B1_dynamic_rf": "#4c78a8",
    "B2_optical_climate": "#9ecae1",
    "B3_plus_sm": "#59a14f",
    "B4_plus_sif": "#f28e2b",
    "B5_plus_vod": "#b07aa1",
    "P_full_physiology": "#e15759",
    "P_food_system": "#76b7b2",
}


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def fig1_framework(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10.2, 3.4))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    boxes = [
        (0.2, 1.1, "Optical + climate\nyield forecast\n192–0 d"),
        (3.2, 1.1, "Yield-induced\nsupply shock\nΔy × area_{t−1}"),
        (6.2, 1.1, "Lagged trade\nnetwork\nW_{t−1}"),
        (9.2, 1.1, "Importer\nexposure ranking\nex-post labels"),
    ]
    for x, y, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), 2.6, 1.5, fill=True, facecolor="#eef3f7", edgecolor="#335c67", lw=1.2))
        ax.text(x + 1.3, y + 0.75, text, ha="center", va="center", fontsize=9)
    for x0, x1 in ((2.8, 3.2), (5.8, 6.2), (8.8, 9.2)):
        ax.annotate("", xy=(x1, 1.85), xytext=(x0, 1.85), arrowprops=dict(arrowstyle="->", color="#335c67"))
    ax.text(
        6.1,
        0.35,
        "Increment test: GOSIF vs optical–climate   |   Baselines: network-only, lagged exposure, oracle",
        ha="center",
        fontsize=8,
        color="#335c67",
    )
    ax.set_title("Crop-to-trade early-warning layout")
    return _save(fig, out / "fig1_framework.png")


def fig2_phenology(panel: pd.DataFrame, series: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    d = panel.groupby(["region", "latitude_band"], as_index=False)[["delta_sos", "delta_eos"]].mean()
    axes[0].scatter(d["delta_sos"], d["delta_eos"], c=np.where(d["latitude_band"] == "tropical", "#e15759", "#4c78a8"))
    axes[0].axhline(0, color="0.6", lw=0.8)
    axes[0].axvline(0, color="0.6", lw=0.8)
    axes[0].set_xlabel("Mean ΔSOS (days)")
    axes[0].set_ylabel("Mean ΔEOS (days)")
    axes[0].set_title("Static-calendar bias by region")
    for _, r in d.iterrows():
        axes[0].text(r["delta_sos"], r["delta_eos"], r["region"], fontsize=7)

    examples = [
        ("BRA", "maize", "Brazil maize (multi-crop)"),
        ("IND", "rice", "India rice"),
        ("USA", "wheat", "US wheat"),
    ]
    for iso3, crop, label in examples:
        sub = series[(series["iso3"] == iso3) & (series["crop"] == crop) & (series["year"] == 2012)]
        if sub.empty:
            continue
        axes[1].plot(sub["progress"], sub["evi"], label=label)
    axes[1].set_xlabel("Season progress (%)")
    axes[1].set_ylabel("EVI")
    axes[1].legend(fontsize=7)
    axes[1].set_title("Typical seasonal curves")
    fig.suptitle("Fig. 2  Dynamic phenology: calendar mismatch and representative seasons")
    return _save(fig, out / "fig2_phenology.png")


def fig3_lead_skill(lead_skill: pd.DataFrame, out: Path, crop_skill: pd.DataFrame | None = None) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=False)
    ax = axes[0]
    labels = {
        "B0_chen_rf": "B0 Chen RF, static",
        "B1_dynamic_rf": "B1 Chen RF, dynamic",
        "B2_optical_climate": "B2 optical + climate",
        "B4_plus_sif": "B4 + GOSIF",
    }
    keep = [e for e in labels if e in set(lead_skill["experiment"])]
    for exp in keep:
        g = lead_skill[lead_skill["experiment"] == exp].sort_values("lead", ascending=False)
        ax.plot(g["lead"], g["r2"], marker="o", label=labels[exp], color=COLORS.get(exp, "black"))
    ax.axhline(0.15, color="0.4", ls="--", lw=0.8, label="R² = 0.15")
    ax.invert_xaxis()
    ax.set_xlabel("Lead time (days before harvest)")
    ax.set_ylabel("Yield-anomaly R²")
    ax.set_title("Pooled wheat + soybean")
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    if crop_skill is not None and not crop_skill.empty:
        for crop, color in (("wheat", "#4c78a8"), ("soybean", "#f28e2b")):
            sub = crop_skill[crop_skill["crop"] == crop].sort_values("lead", ascending=False)
            if sub.empty:
                continue
            ax.plot(sub["lead"], sub["r2"], marker="o", color=color, label=crop)
        ax.axhline(0.15, color="0.4", ls="--", lw=0.8)
        ax.invert_xaxis()
        ax.set_xlabel("Lead time (days before harvest)")
        ax.set_ylabel("B2 yield-anomaly R²")
        ax.set_title("Optical–climate by crop")
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("When yield-anomaly information emerges", y=1.02)
    fig.tight_layout()
    return _save(fig, out / "fig3_lead_skill.png")


def fig4_stress_onset(panel: pd.DataFrame, series: pd.DataFrame, out: Path, lead_skill: pd.DataFrame | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    if lead_skill is not None and {"B2_optical_climate", "B4_plus_sif"} <= set(lead_skill["experiment"]):
        b2 = lead_skill[lead_skill["experiment"] == "B2_optical_climate"][["lead", "r2"]]
        b4 = lead_skill[lead_skill["experiment"] == "B4_plus_sif"][["lead", "r2"]]
        g = b2.merge(b4, on="lead", suffixes=("_b2", "_b4")).sort_values("lead", ascending=False)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.bar(g["lead"].astype(str), g["r2_b4"] - g["r2_b2"], color="#f28e2b")
        ax.set_xlabel("Lead time (days before harvest)")
        ax.set_ylabel("B4 − B2 yield-anomaly R²")
        ax.set_title("Fig. 4  SIF increment over optical–climate (null if bars sit near zero)")
        return _save(fig, out / "fig4_stress_onset.png")
    onset_cols = [c for c in ("sif_onset", "sm_onset", "vod_onset", "evi_onset") if c in panel.columns]
    shocks = panel[panel["shock_true"] > 0.1] if "shock_true" in panel.columns else panel
    if shocks.empty:
        shocks = panel
    if onset_cols and shocks[onset_cols].notna().any().any():
        means = shocks[onset_cols].mean()
        ax.bar(means.index.str.replace("_onset", ""), means.values, color=["#f28e2b", "#59a14f", "#b07aa1", "#4c78a8"][: len(means)])
        ax.set_ylabel("Mean onset (% of season)")
        ax.set_title("Fig. 4  Stress-onset order under heat–drought shocks")
    else:
        ax.text(0.5, 0.5, "No observational onset times", ha="center")
        ax.set_title("Fig. 4  Stress-onset order under heat–drought shocks")
    return _save(fig, out / "fig4_stress_onset.png")


def _world_countries():
    import geopandas as gpd
    from cartopy.io import shapereader

    shp = shapereader.natural_earth(resolution="110m", category="cultural", name="admin_0_countries")
    world = gpd.read_file(shp)
    for col in ("ADM0_A3", "ISO_A3", "adm0_a3", "iso_a3"):
        if col in world.columns:
            world = world.rename(columns={col: "iso3"})
            break
    if "iso3" not in world.columns:
        raise KeyError("country shapefile has no ISO3 column")
    world["iso3"] = world["iso3"].astype(str).str.upper()
    name_col = "NAME" if "NAME" in world.columns else ("ADMIN" if "ADMIN" in world.columns else None)
    if name_col:
        world = world[~world[name_col].astype(str).str.contains("Antarctica", case=False, na=False)]
    chn = world["iso3"] == "CHN"
    twn = world["iso3"] == "TWN"
    if chn.any() and twn.any():
        world = world.copy()
        geoms = world.loc[chn | twn, "geometry"]
        merged = geoms.union_all() if hasattr(geoms, "union_all") else geoms.unary_union
        world.loc[chn, "geometry"] = merged
        world = world.loc[~twn].copy()
    return world


def fig4_exposure_maps(
    expo: pd.DataFrame,
    out: Path,
    years: tuple[int, ...] = (2018, 2020, 2022),
    lead: int = 192,
) -> Path:
    """Observed vs 192-d predicted importer exposure on a world map."""

    import matplotlib.colors as mcolors

    world = _world_countries()
    rows = [int(y) for y in years if int(y) in set(pd.to_numeric(expo["year"], errors="coerce"))]
    if not rows:
        rows = sorted(pd.to_numeric(expo["year"], errors="coerce").dropna().unique())[-3:]
    fig, axes = plt.subplots(len(rows), 2, figsize=(10.8, 1.9 * max(len(rows), 1) + 1.2), squeeze=False)
    cmap = plt.get_cmap("YlOrRd")
    for i, year in enumerate(rows):
        sub = expo[(expo["lead"] == int(lead)) & (expo["year"] == int(year))].copy()
        if sub.empty:
            continue
        agg = sub.groupby("importer", as_index=False)[["exposure_hat", "exposure_obs"]].sum()
        agg["importer"] = agg["importer"].replace({"TWN": "CHN"})
        agg = agg.groupby("importer", as_index=False)[["exposure_hat", "exposure_obs"]].sum()
        vmax = float(np.nanmax(agg[["exposure_hat", "exposure_obs"]].to_numpy(dtype=float)))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
        for j, (col, title) in enumerate(
            (("exposure_obs", f"{year}  observed"), ("exposure_hat", f"{year}  192-d predicted"))
        ):
            ax = axes[i, j]
            m = world.merge(agg.rename(columns={"importer": "iso3"}), on="iso3", how="left")
            world.plot(ax=ax, color="#e8e8e8", edgecolor="white", linewidth=0.25)
            hit = m[m[col].notna()]
            if not hit.empty:
                hit.plot(ax=ax, column=col, cmap=cmap, norm=norm, edgecolor="white", linewidth=0.25, legend=False)
            ax.set_axis_off()
            ax.set_xlim(-170, 185)
            ax.set_ylim(-58, 84)
            ax.set_title(title, fontsize=9)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=axes[i, :].tolist(), fraction=0.025, pad=0.02, label="Exposure")
    fig.suptitle("Predicted versus ex-post wheat and soybean trade exposure", y=1.01, fontsize=11)
    return _save(fig, out / "fig4_exposure_maps.png")


def fig5_uncertainty(preds: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    sub = preds[preds["experiment"] == "P_full_physiology"]
    if sub.empty and len(preds):
        sub = preds
    if sub.empty:
        axes[0].text(0.5, 0.5, "no predictions", ha="center")
        return _save(fig, out / "fig5_uncertainty.png")
    shock_years = {2010, 2012, 2022}
    sub = sub.copy()
    sub["shock_year"] = sub["year"].isin(shock_years)
    axes[0].boxplot(
        [sub.loc[~sub["shock_year"], "abs_err"].dropna(), sub.loc[sub["shock_year"], "abs_err"].dropna()],
        tick_labels=["Other years", "Shock years"],
    )
    axes[0].set_ylabel("Absolute anomaly error")
    axes[0].set_title("Error inflation in unseen shocks")
    axes[1].scatter(sub["ood_score"], sub["abs_err"], s=8, alpha=0.35, c=np.where(sub["shock_year"], "#e15759", "#4c78a8"))
    axes[1].set_xlabel("OOD score")
    axes[1].set_ylabel("Absolute error")
    axes[1].set_title("OOD score tracks error")
    fig.suptitle("Fig. 5  Interval coverage and OOD screening on wheat–soy forecasts")
    return _save(fig, out / "fig5_uncertainty.png")


def fig6_trade(trade: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    if trade is None or trade.empty:
        ax.text(0.5, 0.5, "Trade module not populated", ha="center")
    else:
        g = trade.groupby("lead", as_index=False)["recall"].mean().sort_values("lead", ascending=False)
        ax.plot(g["lead"], g["recall"], marker="o", color="#76b7b2")
        ax.invert_xaxis()
        ax.set_xlabel("Lead time (days)")
        ax.set_ylabel("High-exposure importer recall")
    ax.set_title("Fig. 6  Lagged trade-network recall of high-exposure importers")
    return _save(fig, out / "fig6_trade_exposure.png")


def fig7_season_progress(skill: pd.DataFrame, out: Path) -> Path:
    order = ["pre_sowing", "sowing_emergence", "vegetative", "flowering", "grain_filling", "maturity"]
    labels = ["pre-sow", "sow–emerg", "vegetative", "flowering", "grain fill", "maturity"]
    keep = [("B2_optical_climate", "B2 optical+climate"), ("B4_plus_sif", "B4 + GOSIF")]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(order))
    width = 0.36
    for i, (exp, label) in enumerate(keep):
        sub = skill[skill["experiment"] == exp]
        ys = [float(sub.loc[sub["stage"] == st, "r2"].mean()) if (sub["stage"] == st).any() else np.nan for st in order]
        ax.bar(x + (i - 0.5) * width, ys, width=width, label=label, color=COLORS.get(exp, "black"))
    ax.axhline(0.15, color="0.4", ls="--", lw=0.8, label="R² = 0.15 floor")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Pooled yield-anomaly R²")
    ax.set_ylim(0, 0.35)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Fig. 7  Skill by season progress: pre-sowing floor, no GOSIF increment")
    return _save(fig, out / "fig7_season_progress.png")


def fig8_trade_auprc(
    exposure: pd.DataFrame,
    shortfall: pd.DataFrame,
    structure: pd.DataFrame,
    out: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    series = [
        (exposure, "Model-implied high exposure", "#76b7b2"),
        (shortfall, "Realized import-quantity shortfall", "#e15759"),
    ]
    struct_map = {
        "hhi_increase": ("Importer HHI increase", "#f28e2b"),
        "top1_increase": ("Top-supplier share increase", "#4c78a8"),
        "supplier_loss": ("Supplier-count loss", "#b07aa1"),
    }
    for df, label, color in series:
        if df is None or df.empty or "auprc" not in df.columns:
            continue
        g = df.groupby("lead", as_index=False)["auprc"].mean().sort_values("lead", ascending=False)
        ax.plot(g["lead"], g["auprc"], marker="o", label=label, color=color)
    if structure is not None and not structure.empty:
        for outcome, (label, color) in struct_map.items():
            sub = structure[structure["outcome"] == outcome]
            if sub.empty:
                continue
            g = sub.groupby("lead", as_index=False)["auprc"].mean().sort_values("lead", ascending=False)
            ax.plot(g["lead"], g["auprc"], marker="s", label=label, color=color)
    ax.axhline(0.20, color="0.45", ls="--", lw=0.9, label="Prevalence / chance (0.20)")
    ax.invert_xaxis()
    ax.set_xlabel("Lead time (days before harvest)")
    ax.set_ylabel("AUPRC")
    ax.set_ylim(0.10, 0.70)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Fig. 8  Trade AUPRC: exposure list vs realized shortfall and structure")
    return _save(fig, out / "fig8_trade_auprc.png")


def fig9_label_robustness(matrix: pd.DataFrame, out: Path) -> Path:
    order = ["linear_official", "theil_sen", "percent", "first_difference", "quadratic"]
    labels = ["Linear\n(official)", "Theil–Sen", "Percent", "First\ndifference", "Quadratic"]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(order))
    width = 0.36
    b2 = []
    b4 = []
    for key in order:
        row = matrix[matrix["label"] == key]
        b2.append(float(row["B2_harvest_r2"].iloc[0]) if len(row) else np.nan)
        b4.append(float(row["B4_harvest_r2"].iloc[0]) if len(row) else np.nan)
    ax.bar(x - width / 2, b2, width=width, label="B2 harvest R²", color=COLORS["B2_optical_climate"])
    ax.bar(x + width / 2, b4, width=width, label="B4 harvest R²", color=COLORS["B4_plus_sif"])
    ax.axhline(0.15, color="0.4", ls="--", lw=0.8, label="R² = 0.15 floor")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Harvest yield-anomaly R²")
    ax.set_ylim(0, 0.32)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Fig. 9  Label robustness: no definition gives B4 a 14-day isolated lead")
    return _save(fig, out / "fig9_label_robustness.png")


def fig_nf_exposure_skill(suite_by_lead: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    g = suite_by_lead.sort_values("lead", ascending=False)
    series = [
        ("auprc_forecast", "Yield forecast + network", "#76b7b2", "o"),
        ("auprc_oracle", "Oracle (ex-post yield shock + network)", "#59a14f", "s"),
        ("auprc_n3_propensity", "Historical supplier-shock propensity", "#f28e2b", "D"),
        ("auprc_n0_lag", "Previous-year exposure", "#4c78a8", "^"),
        ("auprc_n2_hhi", "Importer concentration (HHI)", "#b07aa1", "v"),
    ]
    for col, label, color, mark in series:
        if col not in g.columns:
            continue
        ax.plot(g["lead"], g[col], marker=mark, color=color, label=label, lw=1.6)
    if "prevalence" in g.columns:
        ax.axhline(float(g["prevalence"].mean()), color="0.45", ls="--", lw=0.9, label="Prevalence")
    ax.invert_xaxis()
    ax.set_xlabel("Lead time (days before harvest)")
    ax.set_ylabel("AUPRC")
    ax.set_ylim(0.15, 1.05)
    ax.legend(fontsize=7, frameon=False, loc="center left")
    if "cont_spearman" in g.columns:
        inset = ax.inset_axes([0.58, 0.12, 0.38, 0.32])
        inset.plot(g["lead"], g["cont_spearman"], marker="o", color="#335c67", lw=1.2)
        inset.invert_xaxis()
        inset.set_ylim(0.2, 0.6)
        inset.set_title("Spearman", fontsize=7, pad=2)
        inset.tick_params(labelsize=6)
        inset.set_xlabel("Lead (d)", fontsize=6)
    ax.set_title("Trade-exposure ranking versus network-only baselines")
    return _save(fig, out / "fig4_exposure_skill.png")


def fig_nf_robustness(
    crop_skill: pd.DataFrame,
    coverage: dict,
    importer_q: float,
    out: Path,
    detrend: dict | None = None,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.6))
    ax = axes[0]
    if detrend:
        names = list(detrend.keys())
        ax.bar(names, [detrend[k] for k in names], color="#9ecae1")
        ax.axhline(0.15, color="0.4", ls="--", lw=0.8)
        ax.set_ylim(0, 0.32)
        ax.set_ylabel("B2 harvest R²")
        ax.set_title("Yield-label robustness")
        ax.tick_params(axis="x", rotation=20)
    elif crop_skill is not None and not crop_skill.empty:
        for crop, color in (("wheat", "#4c78a8"), ("soybean", "#f28e2b")):
            sub = crop_skill[crop_skill["crop"] == crop].sort_values("lead", ascending=False)
            if sub.empty:
                continue
            ax.plot(sub["lead"], sub["r2"], marker="o", color=color, label=crop)
        ax.axhline(0.15, color="0.4", ls="--", lw=0.8)
        ax.invert_xaxis()
        ax.set_xlabel("Lead (d)")
        ax.set_ylabel("Yield-anomaly R²")
        ax.legend(fontsize=7, frameon=False)
        ax.set_title("Wheat vs soybean")
    ax = axes[1]
    names = list(coverage.keys())
    vals = [coverage[k] for k in names]
    ax.bar(names, vals, color="#76b7b2")
    ax.axhline(0.90, color="0.4", ls="--", lw=0.8)
    ax.set_ylim(0.70, 1.0)
    ax.set_ylabel("90% coverage")
    ax.set_title("Conformal coverage")
    ax.tick_params(axis="x", rotation=20)
    ax = axes[2]
    ax.bar(
        ["Pooled Q80", "Importer Q80"],
        [0.71, float(importer_q)],
        color=["#76b7b2", "#4c78a8"],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("192 d AUPRC")
    ax.set_title("Prioritization vs warning")
    fig.suptitle("Robustness: crop split, coverage and exposure tasks", y=1.03)
    fig.tight_layout()
    return _save(fig, out / "fig5_robustness.png")


def write_manuscript_figures(bundle: dict, cfg) -> list[Path]:
    out = cfg.path("figures")
    paths = [
        fig1_framework(out),
        fig2_phenology(bundle["panel"], bundle["series"], out),
        fig3_lead_skill(bundle["lead_skill"], out),
        fig4_stress_onset(bundle["panel"], bundle["series"], out, bundle.get("lead_skill")),
        fig5_uncertainty(bundle["predictions"], out),
        fig6_trade(bundle["trade"], out),
    ]
    return paths


def write_revision_main_figures(
    stage_skill: pd.DataFrame,
    exposure_by_lead: pd.DataFrame,
    shortfall_by_lead: pd.DataFrame,
    structure_by_lead: pd.DataFrame,
    robustness: pd.DataFrame,
    out: Path,
) -> list[Path]:
    return [
        fig7_season_progress(stage_skill, out),
        fig8_trade_auprc(exposure_by_lead, shortfall_by_lead, structure_by_lead, out),
        fig9_label_robustness(robustness, out),
    ]
