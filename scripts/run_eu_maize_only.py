"""Eurostat NUTS2 maize only — 2003/2018/2022 heat-drought sample."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.config import Config, ensure_dirs, load_config
from cropwarn.data.eurostat_maize import EU_MAIZE_CALENDAR, ISO3_META, load_eu_maize
from cropwarn.data.io import read_table
from cropwarn.data.labels import build_labels
from cropwarn.data.observational import build_unit_rice_block
from cropwarn.experiments.runner import run_experiment_suite
from cropwarn.gee.rice_state import load_eu_maize_series

root = Path(__file__).resolve().parents[1]
base = load_config(root)
raw = dict(base.raw)
raw["study"] = dict(raw["study"])
raw["study"]["headline_crops"] = ["maize"]
raw["paths"] = dict(raw["paths"])
raw["paths"]["results"] = "results/eu_maize"
cfg = Config(raw=raw, root=base.root)
ensure_dirs(cfg)

labels = load_eu_maize(root)
if labels.empty:
    raise SystemExit("run scripts/build_eu_maize.py first")
series = load_eu_maize_series(root)
gosif_path = root / "data_interim" / "gosif_eu_maize.csv"
gosif = pd.read_csv(gosif_path, parse_dates=["date"]) if gosif_path.exists() else pd.DataFrame()
rows, extras, panel = build_unit_rice_block(labels, series, cfg.years, gosif, *EU_MAIZE_CALENDAR, crop="maize")
if panel is None or panel.empty:
    raise SystemExit("empty eu maize panel")
meta = pd.DataFrame([{"iso3": k, **v} for k, v in ISO3_META.items()])
panel = panel.merge(meta, on="iso3", how="left")
panel = build_labels(panel)
panel["shock_true"] = panel["yield_anomaly"]
world = {
    "panel": panel,
    "series": pd.DataFrame(rows),
    "trade": read_table(root / "data_interim" / "obs_trade_network.csv"),
}
print(
    "eu maize panel",
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
