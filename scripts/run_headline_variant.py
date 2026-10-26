"""Re-run Nature Food gates on a cached observational world with a new headline."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.config import Config, ensure_dirs, load_config
from cropwarn.data.io import read_table
from cropwarn.experiments.runner import run_experiment_suite

root = Path(__file__).resolve().parents[1]
crops = [c.strip() for c in sys.argv[1].split(",")] if len(sys.argv) > 1 else ["maize"]
out_name = sys.argv[2] if len(sys.argv) > 2 else "headline_" + "_".join(crops)

base = load_config(root)
raw = dict(base.raw)
raw["study"] = dict(raw["study"])
raw["study"]["headline_crops"] = crops
raw["paths"] = dict(raw["paths"])
raw["paths"]["results"] = f"results/{out_name}"
cfg = Config(raw=raw, root=base.root)
ensure_dirs(cfg)

world = {
    "panel": read_table(root / "data_interim" / "obs_panel_labels.csv"),
    "series": read_table(root / "data_interim" / "obs_season_series.csv"),
    "trade": read_table(root / "data_interim" / "obs_trade_network.csv"),
}
if len(sys.argv) > 3:
    keep = {s.strip() for s in sys.argv[3].split(",") if s.strip()}
    world["panel"] = world["panel"][world["panel"]["iso3"].isin(keep)].copy()
    if "iso3" in world["series"].columns:
        world["series"] = world["series"][world["series"]["iso3"].isin(keep)].copy()
    print("iso3 filter", sorted(keep), flush=True)
print("headline", crops, "panel", len(world["panel"]), "series", len(world["series"]), flush=True)
bundle = run_experiment_suite(world, cfg, demo=True, schemes=["rolling_origin"])
lead = bundle["lead_skill"]
keep = lead[lead["experiment"].isin(["B2_optical_climate", "B4_plus_sif", "P_full_physiology"])]
print(keep[["experiment", "lead", "r2", "earliest_lead"]].to_string())
print("gates", json.dumps(bundle["gates"], indent=2, default=str))
