"""China NBS early-rice only — does canopy track the seasonal label?"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.config import Config, ensure_dirs, load_config
from cropwarn.data.io import read_table
from cropwarn.data.observational import build_observational_world
from cropwarn.experiments.runner import run_experiment_suite

root = Path(__file__).resolve().parents[1]
base = load_config(root)
raw = dict(base.raw)
raw["study"] = dict(raw["study"])
raw["study"]["headline_crops"] = ["rice"]
raw["paths"] = dict(raw["paths"])
raw["paths"]["results"] = "results/china_early_rice"
cfg = Config(raw=raw, root=base.root)
ensure_dirs(cfg)
trade = read_table(root / "data_interim" / "obs_trade_network.csv")
world = build_observational_world(root, cfg.years, cfg.crops, trade=trade)
world["panel"] = world["panel"][world["panel"]["crop"].eq("rice")].copy()
if "unit_id" in world["series"].columns:
    world["series"] = world["series"][world["series"]["unit_id"].astype(str).str.startswith("CHN-")].copy()
print(
    "china rice panel",
    len(world["panel"]),
    "units",
    world["panel"]["unit_id"].nunique(),
    "anomaly",
    world["panel"]["yield_anomaly"].notna().sum(),
    flush=True,
)
bundle = run_experiment_suite(world, cfg, demo=True, schemes=["rolling_origin"])
lead = bundle["lead_skill"]
print(lead[lead["lead"].isin([0, 32, 64, 96, 192])][["experiment", "lead", "r2", "earliest_lead"]].to_string())
print("gates", json.dumps(bundle["gates"], indent=2, default=str))
