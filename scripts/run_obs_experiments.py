"""Official wheat+soy observational suite for the Nature Communications package."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.config import ensure_dirs, load_config
from cropwarn.data.io import read_table, write_table
from cropwarn.data.observational import build_observational_world, observational_inventory
from cropwarn.experiments.runner import run_experiment_suite
from cropwarn.pipeline import write_verdict
from cropwarn.viz.figures import write_manuscript_figures
from cropwarn.viz.tables import write_manuscript_tables

NC_SCHEMES = [
    "rolling_origin",
    "shock_holdout",
    "leave_one_year_out",
    "leave_one_region_out",
]

root = Path(__file__).resolve().parents[1]
cfg = load_config(root)
ensure_dirs(cfg)
trade = read_table(root / "data_interim" / "obs_trade_network.csv")
world = build_observational_world(root, cfg.years, cfg.crops, trade=trade)
write_table(world["panel"], cfg.path("data_interim") / "obs_panel_labels.csv")
write_table(world["series"], cfg.path("data_interim") / "obs_season_series.csv")
print("panel", len(world["panel"]), "series", len(world["series"]), "sif", world["series"]["sif"].notna().mean() if len(world["series"]) else 0)
bundle = run_experiment_suite(world, cfg, demo=True, schemes=NC_SCHEMES)
write_table(bundle["features_static"], cfg.path("features") / "obs_features_static.csv")
write_table(bundle["features_dynamic"], cfg.path("features") / "obs_features_dynamic.csv")
write_manuscript_figures(bundle, cfg)
write_manuscript_tables(bundle, cfg)
inventory = observational_inventory(root, world)
verdict = write_verdict(cfg, bundle, inventory)
print(json.dumps(verdict, indent=2, default=str))
print("gates", json.dumps(bundle["gates"], indent=2, default=str))
