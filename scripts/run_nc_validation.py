"""Extra NC validation schemes. Writes results/nc_validation/ and does not overwrite official gates."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.config import ensure_dirs, load_config
from cropwarn.data.io import read_table
from cropwarn.data.observational import build_observational_world
from cropwarn.experiments.runner import run_experiment_suite

root = Path(__file__).resolve().parents[1]
cfg = load_config(root)
ensure_dirs(cfg)
trade = read_table(root / "data_interim" / "obs_trade_network.csv")
world = build_observational_world(root, cfg.years, cfg.crops, trade=trade)
bundle = run_experiment_suite(
    world,
    cfg,
    demo=True,
    schemes=["rolling_origin", "shock_holdout", "leave_one_year_out", "leave_one_region_out"],
    results_dir=root / "results" / "nc_validation",
)
print("schemes", sorted(bundle["metrics"]["scheme"].unique().tolist()) if len(bundle["metrics"]) else [])
print("gates", json.dumps(bundle["gates"], indent=2, default=str))
