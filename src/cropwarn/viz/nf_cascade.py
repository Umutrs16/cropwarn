"""Nature Food main figures: predictability cascade, not a workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cropwarn.viz.figures import _world_countries

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

TEAL = "#76b7b2"
INK = "#335c67"
ORANGE = "#f28e2b"
RED = "#e15759"
GREY = "#c8c8c8"


def _save_nf(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return path


def _best_network(g: pd.DataFrame) -> np.ndarray:
    cols = [c for c in ("auprc_n0_lag", "auprc_n1_mean", "auprc_n3_propensity") if c in g.columns]
    return np.nanmax(np.vstack([g[c].to_numpy(dtype=float) for c in cols]), axis=0)


def fig1_forecast_vs_network(
    yield_by_lead: pd.DataFrame,
    prod_by_lead: pd.DataFrame,
    perm_yield: pd.DataFrame,
    perm_prod: pd.DataFrame,
    ablation_yield: pd.DataFrame,
    ablation_prod: pd.DataFrame,
    observed_y: float,
    observed_p: float,
    out: Path,
    recovery: float | None = None,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    y = yield_by_lead.sort_values("lead", ascending=False)
    p = prod_by_lead.sort_values("lead", ascending=False)

    ax = axes[0, 0]
    ax.plot(y["lead"], y["auprc_forecast"], marker="o", color=TEAL, label="Yield-induced exposure", lw=1.6, ms=4)
    ax.plot(p["lead"], p["auprc_forecast"], marker="s", color=INK, label="Production-based exposure", lw=1.6, ms=4)
    ax.plot(y["lead"], _best_network(y), color=ORANGE, ls="--", lw=1.2, label="Best yield-induced network")
    ax.plot(p["lead"], _best_network(p), color="#b07aa1", ls=":", lw=1.2, label="Best production network")
    if "prevalence" in y.columns:
        ax.axhline(float(y["prevalence"].mean()), color="0.45", ls="-.", lw=0.8, label="Prevalence")
    ax.invert_xaxis()
    ax.set_xlabel("Lead (days before harvest)")
    ax.set_ylabel("AUPRC")
    ax.set_ylim(0.20, 0.90)
    if recovery is not None and np.isfinite(recovery):
        ax.text(0.02, 0.96, f"Oracle AUPRC = 1.00; 192 d recovery {100 * recovery:.0f}%", transform=ax.transAxes, va="top", fontsize=6.5)
    ax.legend(fontsize=5.8, frameon=False, loc="lower left")
    ax.set_title("a  Co-primary exposure ranking", loc="left")

    ax = axes[0, 1]
    ax.plot(y["lead"], y["auprc_forecast"] - _best_network(y), marker="o", color=TEAL, lw=1.6, ms=4, label="Yield-induced")
    ax.plot(p["lead"], p["auprc_forecast"] - _best_network(p), marker="s", color=INK, lw=1.6, ms=4, label="Production-based")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.invert_xaxis()
    ax.set_xlabel("Lead (days before harvest)")
    ax.set_ylabel("ΔAUPRC vs own best network baseline")
    ax.legend(fontsize=6, frameon=False)
    ax.set_title("b  Forecast-added value", loc="left")

    ax = axes[1, 0]
    dy = pd.to_numeric(perm_yield["auprc_perm"], errors="coerce").dropna()
    dp = pd.to_numeric(perm_prod["auprc_perm"], errors="coerce").dropna()
    ax.hist(dy, bins=28, color=TEAL, alpha=0.45, edgecolor="white", label="Yield-induced null")
    ax.hist(dp, bins=28, color=INK, alpha=0.40, edgecolor="white", label="Production-based null")
    ax.axvline(observed_y, color=TEAL, lw=1.8)
    ax.axvline(observed_p, color=INK, lw=1.8, ls="--")
    ax.set_xlabel("Pooled AUPRC (192 d)")
    ax.set_ylabel("Draws")
    ax.legend(
        fontsize=5.8,
        frameon=False,
        title=f"Observed {observed_y:.2f} / {observed_p:.2f}",
        title_fontsize=6,
    )
    ax.set_title("c  Year × crop shock permutation", loc="left")

    ax = axes[1, 1]
    ay = ablation_yield[ablation_yield["dropped"] != "none"][["dropped", "auprc"]].rename(columns={"auprc": "yield"})
    ap = ablation_prod[ablation_prod["dropped"] != "none"][["dropped", "auprc"]].rename(columns={"auprc": "prod"})
    d = ay.merge(ap, on="dropped", how="inner")
    d = d.sort_values("yield")
    yy = np.arange(len(d))
    ax.hlines(yy, d["prod"], d["yield"], color="0.75", lw=1.0)
    ax.scatter(d["yield"], yy, color=TEAL, s=22, zorder=3, label="Yield-induced")
    ax.scatter(d["prod"], yy, color=INK, s=22, marker="s", zorder=3, label="Production-based")
    if "none" in set(ablation_yield["dropped"]):
        ax.axvline(float(ablation_yield.loc[ablation_yield["dropped"] == "none", "auprc"].iloc[0]), color=TEAL, ls="--", lw=0.8)
    if "none" in set(ablation_prod["dropped"]):
        ax.axvline(float(ablation_prod.loc[ablation_prod["dropped"] == "none", "auprc"].iloc[0]), color=INK, ls=":", lw=0.8)
    ax.set_yticks(yy, d["dropped"])
    ax.set_xlabel("AUPRC after zeroing exporter forecast information")
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    ax.set_title("d  Major-exporter information ablation", loc="left")
    fig.tight_layout()
    return _save_nf(fig, out / "fig1_forecast_vs_network.png")


def fig2_yield_information(crop_skill: pd.DataFrame, ablation: pd.DataFrame, shock_skill: dict, out: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    ax = axes[0, 0]
    for crop, color, marker in (("wheat", "#4c78a8", "o"), ("soybean", ORANGE, "s")):
        sub = crop_skill[crop_skill["crop"] == crop].sort_values("lead", ascending=False)
        if sub.empty:
            continue
        ax.plot(sub["lead"], sub["r2"], marker=marker, color=color, label=crop.capitalize(), lw=1.6, ms=4)
    ax.axhline(0.15, color="0.4", ls="--", lw=0.8)
    ax.invert_xaxis()
    ax.set_ylim(0.10, 0.50)
    ax.set_xlabel("Lead (days before harvest)")
    ax.set_ylabel("Yield-anomaly R²")
    ax.legend(fontsize=6, frameon=False)
    ax.set_title("a  Wheat and soybean", loc="left")

    ax = axes[0, 1]
    labels = {"climate": "Climate only", "optical": "Optical only", "optical_climate": "Optical + climate"}
    colors = {"climate": "#b07aa1", "optical": "#59a14f", "optical_climate": "#4c78a8"}
    if ablation is not None and len(ablation):
        for name, lab in labels.items():
            sub = ablation[ablation["model"] == name].sort_values("lead", ascending=False)
            if sub.empty:
                continue
            ax.plot(sub["lead"], sub["r2"], marker="o", color=colors[name], label=lab, lw=1.5, ms=4)
        ax.axhline(0.15, color="0.4", ls="--", lw=0.8)
        ax.invert_xaxis()
        ax.legend(fontsize=6, frameon=False)
    ax.set_ylim(0.10, 0.50)
    ax.set_xlabel("Lead (days before harvest)")
    ax.set_ylabel("Pooled R²")
    ax.set_title("b  Feature-source ablation", loc="left")

    ax = axes[1, 0]
    if ablation is not None and len(ablation):
        oc = ablation[ablation["model"] == "optical_climate"][["lead", "r2"]].rename(columns={"r2": "oc"})
        gs = ablation[ablation["model"] == "plus_gosif"][["lead", "r2"]].rename(columns={"r2": "gs"})
        d = oc.merge(gs, on="lead")
        d["delta"] = d["gs"] - d["oc"]
        d = d.sort_values("lead", ascending=False)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.bar(d["lead"].astype(str), d["delta"], color=RED, width=0.6)
    ax.set_xlabel("Lead (d)")
    ax.set_ylabel("ΔR² (GOSIF − optical–climate)")
    ax.set_title("c  Fluorescence increment", loc="left")

    ax = axes[1, 1]
    names = ["Sign accuracy", "Severe-neg.\nrecall", "Severe-neg.\nAUPRC"]
    vals = [
        float(shock_skill.get("sign_accuracy", np.nan)),
        float(shock_skill.get("severe_neg_recall", np.nan)),
        float(shock_skill.get("severe_neg_auprc", np.nan)),
    ]
    ax.bar(names, vals, color=[TEAL, INK, ORANGE], width=0.62)
    pi = float(shock_skill.get("severe_neg_prevalence", 0.20))
    ax.axhline(pi, color="0.45", ls="--", lw=0.8, label=f"Severe-shock prevalence {pi:.2f}")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.legend(fontsize=6, frameon=False)
    ax.set_title("d  Downside exporter-shock skill (192 d)", loc="left")
    fig.tight_layout()
    return _save_nf(fig, out / "fig2_yield_information.png")


def fig3_trade_maps(
    expo: pd.DataFrame,
    shocks: pd.DataFrame,
    years: list[int],
    out: Path,
    lead: int = 192,
) -> Path:
    import matplotlib.colors as mcolors
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    world = _world_countries()
    fig = plt.figure(figsize=(8.0, 10.2))
    gs = fig.add_gridspec(
        len(years),
        3,
        width_ratios=[1.55, 1.55, 0.62],
        wspace=0.14,
        hspace=0.32,
        left=0.02,
        right=0.99,
        top=0.94,
        bottom=0.03,
    )
    shock_cmap = plt.get_cmap("RdBu")
    expo_cmap = plt.get_cmap("YlOrRd")
    letters = "abcdefghijklmnopqrstuvwxyz"
    border = dict(linewidth=0.70, edgecolor="black")
    for i, year in enumerate(years):
        sh = shocks[shocks["year"] == int(year)].copy()
        if "iso3" in sh.columns and "shock_true" in sh.columns:
            smap = sh.groupby("iso3", as_index=False).agg(shock_true=("shock_true", "mean"))
        else:
            smap = pd.DataFrame(columns=["iso3", "shock_true"])
        sub = expo[(expo["lead"] == int(lead)) & (expo["year"] == int(year))].copy()
        sub["importer"] = sub["importer"].replace({"TWN": "CHN"})
        agg = sub.groupby("importer", as_index=False)[["exposure_hat", "exposure_obs"]].sum()
        if len(agg):
            agg["rank_hat"] = agg["exposure_hat"].rank(ascending=False, method="average")
            agg["rank_obs"] = agg["exposure_obs"].rank(ascending=False, method="average")
            mean_abs_rank = float((agg["rank_hat"] - agg["rank_obs"]).abs().mean())
            top = agg.nsmallest(8, "rank_obs").copy()
        else:
            mean_abs_rank = float("nan")
            top = agg
        vmax_s = float(np.nanpercentile(np.abs(smap["shock_true"]), 95)) if len(smap) else 0.2
        vmax_s = max(vmax_s, 0.05)
        vmax_e = float(np.nanpercentile(agg["exposure_hat"], 95)) if len(agg) else 0.1
        vmax_e = max(vmax_e, 1e-6)

        map_specs = [
            (smap.rename(columns={"shock_true": "val"}), shock_cmap, -vmax_s, vmax_s, "Production-based exporter shock"),
            (agg.rename(columns={"importer": "iso3", "exposure_hat": "val"}), expo_cmap, 0.0, vmax_e, "192-d predicted exposure"),
        ]
        for j, (tab, cmap, vmin, vmax, title) in enumerate(map_specs):
            ax = fig.add_subplot(gs[i, j])
            ax.set_axis_off()
            lookup = tab.dropna(subset=["iso3"]).drop_duplicates("iso3")
            lookup = pd.Series(lookup["val"].to_numpy(), index=lookup["iso3"].astype(str))
            merged = world.copy()
            merged["mapval"] = pd.to_numeric(merged["iso3"].astype(str).map(lookup), errors="coerce")
            merged.plot(ax=ax, color=GREY, **border)
            hit = merged[merged["mapval"].notna()].copy()
            if len(hit):
                hit.plot(ax=ax, column="mapval", cmap=cmap, vmin=vmin, vmax=vmax, **border)
            ax.set_xlim(-170, 185)
            ax.set_ylim(-70, 84)
            ax.set_title(f"{letters[i * 3 + j]}  {year}  {title}", fontsize=7.2, loc="left", pad=2)
            if i == 0 and j == 0:
                ax.text(
                    0.08,
                    0.34,
                    "Grey: no data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    color="0.15",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9),
                )
            sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin, vmax), cmap=cmap)
            sm.set_array([])
            cax = inset_axes(ax, width="62%", height="4.2%", loc="lower center", borderpad=0.15)
            cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=5.5, pad=1)
            cb.set_label("Shock" if j == 0 else "Exposure", fontsize=6)

        ax = fig.add_subplot(gs[i, 2])
        ax.set_xlim(-0.42, 1.42)
        if len(top):
            top = top.copy()
            top["slot_hat"] = top["rank_hat"].rank(method="first")
            top["slot_obs"] = top["rank_obs"].rank(method="first")
            ax.set_ylim(8.7, 0.3)
            for _, r in top.iterrows():
                code = str(r["importer"])
                sh, so = float(r["slot_hat"]), float(r["slot_obs"])
                rh, ro = float(r["rank_hat"]), float(r["rank_obs"])
                ax.plot([0, 1], [sh, so], color=INK, lw=1.05, alpha=0.85)
                ax.scatter([0, 1], [sh, so], s=16, c=INK, zorder=3)
                ax.text(-0.07, sh, f"{code}  {rh:.0f}", ha="right", va="center", fontsize=6.2)
                ax.text(1.07, so, f"{ro:.0f}  {code}", ha="left", va="center", fontsize=6.2)
        ax.set_xticks([0, 1], ["Predicted", "Ex-post"], fontsize=7)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.spines["left"].set_visible(False)
        rank_note = f"\nmean |Δrank| = {mean_abs_rank:.1f}" if np.isfinite(mean_abs_rank) else ""
        ax.set_title(f"{letters[i * 3 + 2]}  {year}  Rank agreement{rank_note}", fontsize=7.2, loc="left", pad=2)
    fig.suptitle("Forecast ranking of network-implied wheat and soybean trade exposure", y=0.99, fontsize=10)
    return _save_nf(fig, out / "fig3_trade_maps.png")


def fig4_attenuation(forest: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    d = forest.copy().reset_index(drop=True)
    y = np.arange(len(d))[::-1]
    lo = d["p025"] if "p025" in d.columns else d["p05"]
    hi = d["p975"] if "p975" in d.columns else d["p95"]
    colors = [TEAL if str(t) == "exposure" else RED for t in d["layer"]]
    ax.axvline(0, color="0.45", lw=1.0)
    n_exp = int((d["layer"] == "exposure").sum())
    if 0 < n_exp < len(d):
        gap_y = (y[n_exp - 1] + y[n_exp]) / 2
        ax.axhline(gap_y, color="0.82", lw=0.8, ls="--")
    ax.hlines(y, lo, hi, color="0.35", lw=1.2)
    ax.scatter(d["skill"], y, c=colors, s=36, zorder=3, edgecolors="white", linewidths=0.4)
    ax.set_yticks(y, d["label"])
    ax.set_xlabel("Average-precision skill relative to prevalence  S = (AUPRC − π)/(1 − π)")
    ax.set_xlim(-0.15, 0.85)
    ax.set_title("Predictability gap between physical exposure and trade adjustment", loc="left")
    return _save_nf(fig, out / "fig4_attenuation.png")


def fig5_buffers(split: pd.DataFrame, grid: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    mapping = [
        (axes[0, 0], "idr_lag1", "a  Import dependence (IDR 0.5)"),
        (axes[0, 1], "hhi_lag1", "b  Supplier concentration"),
        (axes[1, 0], "n_suppliers_lag1", "c  Supplier count"),
    ]
    port = split[split.get("score", "portfolio") == "portfolio"] if "score" in split.columns else split
    for ax, col, title in mapping:
        sub = port[port["split"] == col]
        bars = sub[sub["stratum"].isin(["low", "high"])]
        if bars.empty:
            ax.set_title(title, fontsize=9)
            continue
        ax.bar(bars["stratum"], bars["auprc"], color=["#9ecae1", TEAL])
        if "prevalence" in bars.columns:
            ax.axhline(float(bars["prevalence"].mean()), color="0.45", ls="--", lw=0.8)
        ax.set_ylim(0, 0.55)
        ax.set_ylabel("AUPRC vs import shortfall", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
    ax = axes[1, 1]
    if grid is not None and len(grid):
        mat = np.full((2, 2), np.nan)
        idx = {"low": 0, "high": 1}
        for _, r in grid.iterrows():
            mat[idx[str(r["row"])], idx[str(r["col"])]] = r["p_shortfall_given_high_expo"]
        im = ax.imshow(mat, vmin=0, vmax=0.6, cmap="YlOrRd")
        ax.set_xticks([0, 1], ["Low HHI", "High HHI"], fontsize=8)
        ax.set_yticks([0, 1], ["Low IDR", "High IDR"], fontsize=8)
        for i in range(2):
            for j in range(2):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        cb.set_label("P(shortfall | high exposure)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    ax.set_title("d  Dependence × concentration", fontsize=9)
    fig.tight_layout()
    return _save_nf(fig, out / "fig5_buffers.png")
