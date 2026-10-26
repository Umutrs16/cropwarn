"""US NASS state maize only — smoother label, same 2012-belt RS."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.config import Config, ensure_dirs, load_config
from cropwarn.constants import COUNTRIES
from cropwarn.data.io import read_table
from cropwarn.data.labels import build_labels
from cropwarn.data.nass_maize import MAIZE_CALENDAR, load_usa_maize_state, rollup_to_state
from cropwarn.data.observational import build_unit_rice_block
from cropwarn.experiments.runner import run_experiment_suite
from cropwarn.gee.rice_state import load_usa_maize_series

root = Path(__file__).resolve().parents[1]
base = load_config(root)
raw = dict(base.raw)
raw["study"] = dict(raw["study"])
raw["study"]["headline_crops"] = ["maize"]
raw["paths"] = dict(raw["paths"])
raw["paths"]["results"] = "results/usa_maize_state"
cfg = Config(raw=raw, root=base.root)
ensure_dirs(cfg)

labels = load_usa_maize_state(root)
if labels.empty:
    raise SystemExit("run scripts/build_usa_maize_state.py first")
series = rollup_to_state(load_usa_maize_series(root))
gosif_path = root / "data_interim" / "gosif_usa_maize_state.csv"
if gosif_path.exists():
    gosif = pd.read_csv(gosif_path, parse_dates=["date"])
else:
    gosif = rollup_to_state(pd.read_csv(root / "data_interim" / "gosif_usa_maize.csv", parse_dates=["date"]))
rows, extras, panel = build_unit_rice_block(labels, series, cfg.years, gosif, *MAIZE_CALENDAR, crop="maize")
if panel is None or panel.empty:
    raise SystemExit("empty state maize panel")
meta = pd.DataFrame([asdict(c) for c in COUNTRIES])
panel = panel.merge(meta, on="iso3", how="left")
if panel["harvested_ha"].isna().all():
    panel["harvested_ha"] = 3.0e6
panel = build_labels(panel)
panel["shock_true"] = panel["yield_anomaly"]
world = {
    "panel": panel,
    "series": pd.DataFrame(rows),
    "trade": read_table(root / "data_interim" / "obs_trade_network.csv"),
}
print(
    "state maize panel",
    len(world["panel"]),
    "units",
    world["panel"]["unit_id"].nunique(),
    "anomaly",
    world["panel"]["yield_anomaly"].notna().sum(),
    "series",
    len(world["series"]),
    flush=True,
)
bundle = run_experiment_suite(world, cfg, demo=True, schemes=["rolling_origin"])
lead = bundle["lead_skill"]
print(lead[lead["lead"].isin([0, 32, 64, 96, 192])][["experiment", "lead", "r2", "earliest_lead"]].to_string())
print("gates", json.dumps(bundle["gates"], indent=2, default=str))
