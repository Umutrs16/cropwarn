"""Stage 0–12 orchestration matching section 5.3 of the paper framework."""

from __future__ import annotations

from pathlib import Path

from cropwarn.config import Config, ensure_dirs, load_config
from cropwarn.data.external import write_external_readme
from cropwarn.data.io import write_table
from cropwarn.data.observational import build_observational_world
from cropwarn.data.synthetic import generate_world
from cropwarn.experiments.runner import run_experiment_suite
from cropwarn.validation.journal import recommend_journal
from cropwarn.validation.leakage import assert_realtime_phenology
from cropwarn.viz.figures import write_manuscript_figures
from cropwarn.viz.tables import write_manuscript_tables


STAGES = [
    "0_reproduce_baseline_contract",
    "1_labels",
    "2_crop_masks",
    "3_dynamic_phenology",
    "4_gee_feature_factory",
    "5_sif_vod_fusion",
    "6_baseline_vs_dynamic_phenology",
    "7_physiology_ablation",
    "8_uncertainty",
    "9_shock_holdout",
    "10_trade_exposure",
    "11_subnational_hooks",
    "12_robustness_and_figures",
]


def stage_notes() -> dict[str, str]:
    return {
        "0_reproduce_baseline_contract": "Chen-style clustered RF is Baseline 1. Metrics and leads stay comparable.",
        "1_labels": "Rolling trend uses only years <= t-1. Primary target is yield anomaly.",
        "2_crop_masks": "CROPGRIDS/SPAM weights for the long run; WorldCereal 2021 is a check, not yearly truth.",
        "3_dynamic_phenology": "static / real-time dynamic / oracle versions. Early EOS is a historical prior.",
        "4_gee_feature_factory": "gee/*.js export country statistics up to the forecast cutoff only.",
        "5_sif_vod_fusion": "Offline GOSIF/VODCA join on country-crop-year-date with availability masks.",
        "6_baseline_vs_dynamic_phenology": "B0 vs B1 isolates the phenology module.",
        "7_physiology_ablation": "Optical → +SM → +SIF → +VOD → full on identical folds and leads.",
        "8_uncertainty": "Split conformal 90% intervals plus Mahalanobis/kNN/energy OOD.",
        "9_shock_holdout": "Whole event years held out. Score both error and whether OOD fired.",
        "10_trade_exposure": "Lagged FAOSTAT/BACI network. Claims are risk information, not policy causality.",
        "11_subnational_hooks": "US NASS / Eurostat maize and China early rice were extracted and held out of headline.",
        "12_robustness_and_figures": "Official features drop country ID. Quadratic detrend is the executed sensitivity.",
    }


def run_demo(root: str | Path | None = None) -> dict:
    cfg = load_config(root)
    ensure_dirs(cfg)
    write_external_readme(cfg)

    world = generate_world(cfg.years, cfg.crops, seed=cfg.seed)
    write_table(world["panel"], cfg.path("data_interim") / "panel_labels.csv")
    write_table(world["series"], cfg.path("data_interim") / "season_series.csv")
    write_table(world["trade"], cfg.path("data_interim") / "trade_network.csv")
    write_table(world["countries"], cfg.path("data_interim") / "countries.csv")

    bundle = run_experiment_suite(world, cfg, demo=True)
    write_table(bundle["features_static"], cfg.path("features") / "features_static.csv")
    write_table(bundle["features_dynamic"], cfg.path("features") / "features_dynamic.csv")
    assert_realtime_phenology(bundle["features_dynamic"])

    figs = write_manuscript_figures(bundle, cfg)
    tabs = write_manuscript_tables(bundle, cfg)
    _write_stage_log(cfg)
    bundle["figures"] = figs
    bundle["tables"] = tabs
    bundle["config"] = cfg
    return bundle


def write_verdict(cfg: Config, bundle: dict, inventory: dict) -> dict:
    import json

    verdict = recommend_journal(bundle["gates"], inventory)
    (cfg.path("results") / "journal_verdict.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    md = [f"# Journal recommendation: {verdict['recommendation']}", "", verdict["reason"], ""]
    md.extend(f"- {n}" for n in verdict["notes"])
    (cfg.path("results") / "journal_verdict.md").write_text("\n".join(md), encoding="utf-8")
    return verdict


def run_observational(root: str | Path | None = None) -> dict:
    """FAOSTAT labels + national cropland-weighted GEE + GOSIF. Not the demo."""

    cfg = load_config(root)
    ensure_dirs(cfg)
    from cropwarn.data.gosif import download_gosif_monthly, sample_gosif_national
    from cropwarn.data.observational import observational_inventory, parse_trade_zip
    from cropwarn.data.io import read_table
    from cropwarn.gee.national import extract_all_national, sample_cropland_points

    extract_all_national(cfg.root, years=cfg.years)
    try:
        sample_cropland_points(cfg.root)
    except Exception as exc:
        print(f"cropland points fallback: {exc}", flush=True)
    download_gosif_monthly(cfg.root, cfg.years)
    sample_gosif_national(cfg.root, cfg.years)

    trade_csv = cfg.path("data_interim") / "obs_trade_network.csv"
    tm = cfg.path("data_raw") / "Trade_DetailedTradeMatrix_E_All_Data_(Normalized).zip"
    trade = read_table(trade_csv) if trade_csv.exists() else (parse_trade_zip(tm) if tm.exists() else None)
    world = build_observational_world(cfg.root, cfg.years, cfg.crops, trade=trade)
    write_table(world["panel"], cfg.path("data_interim") / "obs_panel_labels.csv")
    write_table(world["series"], cfg.path("data_interim") / "obs_season_series.csv")
    if world["trade"] is not None and len(world["trade"]):
        write_table(world["trade"], cfg.path("data_interim") / "obs_trade_network.csv")

    bundle = run_experiment_suite(
        world,
        cfg,
        demo=True,
        schemes=["rolling_origin", "shock_holdout", "leave_one_year_out", "leave_one_region_out"],
    )
    write_table(bundle["features_static"], cfg.path("features") / "obs_features_static.csv")
    write_table(bundle["features_dynamic"], cfg.path("features") / "obs_features_dynamic.csv")
    figs = write_manuscript_figures(bundle, cfg)
    tabs = write_manuscript_tables(bundle, cfg)
    inventory = observational_inventory(cfg.root, world)
    verdict = write_verdict(cfg, bundle, inventory)
    _write_stage_log(cfg)
    bundle["figures"] = figs
    bundle["tables"] = tabs
    bundle["verdict"] = verdict
    bundle["inventory"] = inventory
    bundle["config"] = cfg
    return bundle


def run_full(root: str | Path | None = None) -> dict:
    cfg = load_config(root)
    ensure_dirs(cfg)
    world = generate_world(cfg.years, cfg.crops, seed=cfg.seed)
    bundle = run_experiment_suite(world, cfg, demo=False)
    write_manuscript_figures(bundle, cfg)
    write_manuscript_tables(bundle, cfg)
    _write_stage_log(cfg)
    return bundle


def _write_stage_log(cfg: Config) -> None:
    notes = stage_notes()
    lines = ["# Pipeline stages (paper §5.3)", ""]
    for i, name in enumerate(STAGES):
        lines.append(f"## Stage {name}")
        lines.append(notes[name])
        lines.append("")
    (cfg.path("results") / "pipeline_stages.md").write_text("\n".join(lines), encoding="utf-8")
