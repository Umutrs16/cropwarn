"""Agricultural Systems figures: biophysical diagnosis, coupling, constrained value."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from cropwarn.viz.nf_cascade import TEAL, INK, ORANGE, RED, GREY, fig3_trade_maps, _save_nf

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


def fig1_framework(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    boxes = [
        (0.2, 3.4, 2.2, 1.8, "#d9edec", "Phenology\npre-sowing to\nmaturity"),
        (2.7, 3.4, 2.3, 1.8, "#dce8e4", "Observables shrink\nwith lead: meteo–\nthermal, then canopy"),
        (5.3, 3.4, 2.3, 1.8, "#cfe2e0", "Forecast: yield shock\nLabel: production\nresidual"),
        (7.9, 3.4, 1.9, 1.8, "#b7d4d1", "Exporter\nshock"),
        (10.1, 3.4, 1.7, 1.8, TEAL, "Importer\nexposure"),
        (0.2, 0.5, 3.5, 1.8, "#f4e4d4", "192-d forecast\n(often pre-sowing)"),
        (4.0, 0.5, 3.8, 1.8, "#ead8c8", "Lagged t−1 / t−2\ntrade network"),
        (8.1, 0.5, 3.6, 1.8, "#f2d6d6", "Top-K monitoring\nbudget; not a loss\nforecast"),
    ]
    for x, y, w, h, c, t in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", facecolor=c, edgecolor="0.25", linewidth=0.8))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.2)
    for x0, x1 in ((2.4, 2.7), (5.0, 5.3), (7.6, 7.9), (9.8, 10.1)):
        ax.annotate("", xy=(x1, 4.3), xytext=(x0, 4.3), arrowprops=dict(arrowstyle="->", color="0.3", lw=1.0))
    ax.annotate("", xy=(3.8, 3.4), xytext=(2.0, 2.35), arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0))
    ax.annotate("", xy=(8.8, 3.4), xytext=(5.9, 2.35), arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0))
    ax.annotate("", xy=(9.9, 2.35), xytext=(10.9, 3.4), arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0))
    ax.text(0.35, 4.95, "a", fontsize=9, fontweight="bold", va="top")
    ax.text(0.35, 2.05, "b", fontsize=9, fontweight="bold", va="top")
    return _save_nf(fig, out / "Figure_1.png")


def fig2_biophysical(
    crop_skill: pd.DataFrame,
    stage: pd.DataFrame,
    strata: pd.DataFrame,
    shock: dict,
    ablation: pd.DataFrame,
    out: Path,
    source_stage: pd.DataFrame | None = None,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    ax = axes[0, 0]
    for crop, color, marker in (("wheat", "#4c78a8", "o"), ("soybean", ORANGE, "s")):
        sub = crop_skill[crop_skill["crop"] == crop].sort_values("lead", ascending=False)
        if sub.empty:
            continue
        ax.plot(sub["lead"], sub["r2"], marker=marker, color=color, label=crop.capitalize(), lw=1.6, ms=4)
    ax.axhline(0.15, color="0.4", ls="--", lw=0.8)
    ax.invert_xaxis()
    ax.set_ylim(-0.05, 0.55)
    ax.set_xlabel("Lead (days before harvest)")
    ax.set_ylabel("Yield-anomaly R²")
    ax.legend(fontsize=6, frameon=False)
    ax.set_title("a  Crop-specific yield skill", loc="left")

    ax = axes[0, 1]
    order = ["pre_sowing", "sowing_emergence", "vegetative", "flowering", "grain_filling", "maturity"]
    labels = ["Pre-sowing", "Sowing", "Vegetative", "Flowering", "Grain fill", "Maturity"]
    if source_stage is not None and len(source_stage):
        specs = (
            ("climate", INK, "Met.–thermal"),
            ("optical", ORANGE, "Optical"),
            ("optical_climate", TEAL, "Both"),
        )
        x = np.arange(len(order))
        width = 0.26
        plotted = []
        for i, (model, color, lab) in enumerate(specs):
            sub = source_stage[source_stage["model"] == model]
            vals = [float(sub.loc[sub["stage"] == s, "r2"].iloc[0]) if s in set(sub["stage"]) else np.nan for s in order]
            plotted.extend(vals)
            ax.bar(x + (i - 1) * width, vals, width=width, color=color, label=lab)
        ax.set_xticks(x, labels, rotation=25, fontsize=6.5)
        finite = [v for v in plotted if np.isfinite(v)]
        lo = min(-0.02, min(finite) - 0.02) if finite else -0.02
        hi = max(0.50, max(finite) + 0.18) if finite else 0.50
        ax.set_ylim(lo, hi)
        ax.legend(fontsize=5.8, frameon=False, loc="upper left")
        ax.set_title("b  Source of skill by stage", loc="left")
    else:
        st = stage[stage["experiment"] == "B2_optical_climate"].copy()
        vals = [float(st.loc[st["stage"] == s, "r2"].iloc[0]) if s in set(st["stage"]) else np.nan for s in order]
        ax.bar(labels, vals, color=TEAL, width=0.7)
        ax.tick_params(axis="x", rotation=25, labelsize=6.5)
        ax.set_title("b  Skill by phenological stage", loc="left")
    ax.axhline(0, color="0.45", lw=0.7)
    ax.set_ylabel("Yield-anomaly R²")

    ax = axes[1, 0]
    band = strata[(strata["stratum"] == "climate_band") & (strata["lead"].isin([192, 96, 0]))]
    colors = {"tropical": RED, "subtropical": TEAL, "temperate": INK}
    for level, g in band.groupby("level"):
        g = g.sort_values("lead", ascending=False)
        ax.plot(g["lead"], g["r2"], marker="o", color=colors.get(level, GREY), label=level.capitalize(), lw=1.5, ms=4)
    ax.axhline(0, color="0.5", lw=0.7)
    ax.invert_xaxis()
    ax.set_xlabel("Lead (days before harvest)")
    ax.set_ylabel("Yield-anomaly R²")
    ax.legend(fontsize=6, frameon=False)
    ax.set_title("c  Latitude-band performance", loc="left")

    ax = axes[1, 1]
    names = ["Sign\naccuracy", "Severe-neg.\nrecall", "Severe-neg.\nAUPRC"]
    vals = [float(shock.get("sign_accuracy", np.nan)), float(shock.get("severe_neg_recall", np.nan)), float(shock.get("severe_neg_auprc", np.nan))]
    ax.bar(names, vals, color=[TEAL, INK, ORANGE], width=0.62)
    prevalence = float(shock.get("severe_neg_prevalence", 0.20))
    ax.hlines(0.50, -0.28, 0.28, color="0.2", lw=1.05)
    ax.hlines(prevalence, 1.72, 2.28, color="0.2", lw=1.05, ls="--")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("d  Severe exporter-shock recognition (192 d)", loc="left")
    ax.text(0.02, 0.98, "Ticks: sign chance 0.50; severe prevalence (AUPRC only)", transform=ax.transAxes, va="top", fontsize=5.6, color="0.25")
    fig.tight_layout()
    return _save_nf(fig, out / "Figure_2.png")


def fig3_endpoints(y_lead, p_lead, y_sum, p_sum, perm_y, perm_p, y2, p2, observed_y, observed_p, out: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    y = y_lead.sort_values("lead", ascending=False)
    p = p_lead.sort_values("lead", ascending=False)
    ax = axes[0, 0]
    ax.plot(y["lead"], y["auprc_forecast"], marker="o", color=TEAL, label="Yield-induced", lw=1.6, ms=4)
    ax.plot(p["lead"], p["auprc_forecast"], marker="s", color=INK, label="Production-based", lw=1.6, ms=4)
    ax.plot(y["lead"], y["auprc_n0_lag"], color=ORANGE, ls="--", lw=1.1, label="Yield N0")
    ax.plot(p["lead"], p["auprc_n0_lag"], color="#b07aa1", ls=":", lw=1.1, label="Production N0")
    ax.invert_xaxis()
    ax.set_ylim(0.20, 0.90)
    ax.set_xlabel("Lead (days before harvest)")
    ax.set_ylabel("AUPRC")
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    ax.set_title("a  Co-primary exposure ranking", loc="left")

    ax = axes[0, 1]
    rows = [
        ("Yield − N0", y_sum["delta_n0"], y_sum["delta_n0_p025"], y_sum["delta_n0_p975"], TEAL),
        ("Yield − best net.", y_sum["delta_best"], y_sum["delta_best_p025"], y_sum["delta_best_p975"], TEAL),
        ("Prod. − N0", p_sum["delta_n0"], p_sum["delta_n0_p025"], p_sum["delta_n0_p975"], INK),
        ("Prod. − best net.", p_sum["delta_best"], p_sum["delta_best_p025"], p_sum["delta_best_p975"], INK),
        ("Yield, W t−2", y2["delta_n0"], y2["delta_n0_p025"], y2["delta_n0_p975"], "#59a14f"),
        ("Prod., W t−2", p2["delta_n0"], p2["delta_n0_p025"], p2["delta_n0_p975"], "#59a14f"),
    ]
    yy = np.arange(len(rows))[::-1]
    ax.axvline(0, color="0.45", lw=1.0)
    for i, (lab, m, lo, hi, c) in enumerate(rows):
        ax.hlines(yy[i], lo, hi, color="0.35", lw=1.2)
        ax.scatter(m, yy[i], color=c, s=28, zorder=3)
    ax.set_yticks(yy, [r[0] for r in rows], fontsize=6.5)
    ax.set_xlabel("Paired year-block ΔAUPRC")
    ax.set_title("b  Increment versus network rules", loc="left")

    ax = axes[1, 0]
    dy = pd.to_numeric(perm_y["auprc_perm"], errors="coerce").dropna()
    dp = pd.to_numeric(perm_p["auprc_perm"], errors="coerce").dropna()
    ax.hist(dy, bins=28, color=TEAL, alpha=0.45, edgecolor="white", label="Yield null")
    ax.hist(dp, bins=28, color=INK, alpha=0.40, edgecolor="white", label="Production null")
    ax.axvline(observed_y, color=TEAL, lw=1.8)
    ax.axvline(observed_p, color=INK, lw=1.8, ls="--")
    ax.set_xlabel("Year-mean AUPRC (192 d)")
    ax.set_ylabel("Draws")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ymax = ax.get_ylim()[1]
    ax.text(observed_y, ymax * 0.97, f"{observed_y:.2f}", color=TEAL, ha="right", va="top", fontsize=6.2, rotation=90)
    ax.text(observed_p, ymax * 0.97, f"{observed_p:.2f}", color=INK, ha="left", va="top", fontsize=6.2, rotation=90)
    ax.set_title("c  Year-mean shock permutation", loc="left")

    ax = axes[1, 1]
    labels = ["Yield-induced", "Production-based"]
    svals = [y_sum["s_formula"], p_sum["s_formula"]]
    slo = [y_sum["s_p025"], p_sum["s_p025"]]
    shi = [y_sum["s_p975"], p_sum["s_p975"]]
    xs = np.arange(2)
    ax.axhline(0, color="0.45", lw=1.0)
    ax.vlines(xs, slo, shi, color="0.35", lw=1.4)
    ax.scatter(xs, svals, c=[TEAL, INK], s=36, zorder=3)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("S = (AUPRC − π)/(1 − π)")
    ax.set_title("d  Formula-consistent skill", loc="left")
    fig.tight_layout()
    return _save_nf(fig, out / "Figure_3.png")


def fig5_topk(tk: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, endpoint, title in (
        (axes[0], "production", "a  Production-based exposure"),
        (axes[1], "yield", "b  Yield-induced exposure"),
    ):
        g = tk[tk["endpoint"] == endpoint]
        for score, color, lab, ls in (
            ("exposure_hat", TEAL, "Forecast + network", "-"),
            ("n0_lag", ORANGE, "Lagged exposure (N0)", "--"),
            ("n3_propensity", "#b07aa1", "Supplier propensity (N3)", ":"),
        ):
            s = g[g["score"] == score].sort_values("k")
            ax.plot(s["k"], s["captured_obs_share"], color=color, ls=ls, marker="o", ms=4, lw=1.5, label=lab)
            ax.fill_between(s["k"], s["captured_p025"], s["captured_p975"], color=color, alpha=0.12)
        ax.set_xlabel("Monitoring budget K (importers / crop-year)")
        ax.set_ylabel("Share of ex-post exposure captured")
        ax.set_ylim(0.05, 0.90)
        ax.set_title(title, loc="left")
        ax.legend(fontsize=5.8, frameon=False)
    fig.tight_layout()
    return _save_nf(fig, out / "Figure_5.png")


def fig6_downstream(forest: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    d = forest.copy().reset_index(drop=True)
    y = np.arange(len(d))[::-1]
    colors = [TEAL if str(t) == "exposure" else RED for t in d["layer"]]
    ax.axvline(0, color="0.45", lw=1.0)
    ax.hlines(y, d["p025"], d["p975"], color="0.35", lw=1.2)
    ax.scatter(d["skill"], y, c=colors, s=36, zorder=3, edgecolors="white", linewidths=0.4)
    ax.set_yticks(y, d["label"])
    ax.set_xlabel("S = (AUPRC − π)/(1 − π)")
    ax.set_xlim(-0.15, 0.85)
    ax.set_title("Physical exposure versus import-adjustment labels", loc="left")
    return _save_nf(fig, out / "Figure_6.png")


def graphical_abstract(y_sum, p_sum, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13.28, 5.31))
    ax.set_xlim(0, 13.28)
    ax.set_ylim(0, 5.31)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    ax.text(0.35, 4.85, "When do pre-harvest crop forecasts inform agricultural supply-system risk?", fontsize=14, fontweight="bold", va="top")
    ax.text(0.35, 4.35, "Wheat and soybean  ·  rolling-origin 2016–2024  ·  192-day lead", fontsize=9, color="0.3")
    cards = [
        (0.35, 1.15, 3.9, 2.85, TEAL, "Production layer", "Severe negative\nexporter shocks recalled\nat 0.77\nTropical transfer fails"),
        (4.55, 1.15, 4.2, 2.85, INK, "Coupled exposure", f"Yield-induced AUPRC {y_sum['auprc']:.2f}\nS = {y_sum['s_formula']:.2f}\nProduction AUPRC {p_sum['auprc']:.2f}\nS = {p_sum['s_formula']:.2f}"),
        (9.05, 1.15, 3.85, 2.85, RED, "Decision boundary", "Top-K ranking adds\nyield-induced value\nImport adjustment\nremains near chance"),
    ]
    for x, y, w, h, c, title, body in cards:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12", facecolor="white", edgecolor=c, linewidth=2.0))
        ax.add_patch(mpatches.Rectangle((x, y + h - 0.55), w, 0.55, facecolor=c, edgecolor=c))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax.text(x + 0.2, y + h - 0.75, body, ha="left", va="top", fontsize=10)
    ax.text(0.35, 0.45, "Not a forecast of realized import losses or food-access outcomes.", fontsize=9, style="italic", color="0.25")
    dest = out / "Graphical_Abstract.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=100, facecolor="white")
    fig.savefig(dest.with_suffix(".pdf"), facecolor="white")
    fig.savefig(out / "Graphical_Abstract.tiff", dpi=300, facecolor="white")
    plt.close(fig)
    return dest
