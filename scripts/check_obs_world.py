from pathlib import Path

import pandas as pd

from cropwarn.data.io import read_table
from cropwarn.data.observational import build_observational_world

root = Path(r"C:\Users\3SOBS\Desktop\NatureFood")
trade = read_table(root / "data_interim" / "obs_trade_network.csv")
world = build_observational_world(root, list(range(2001, 2024)), ["wheat", "rice", "maize", "soybean"], trade=trade)
print("panel", world["panel"].shape, world["panel"]["yield_tha"].describe())
print("series", world["series"].shape)
print("countries", world["panel"]["iso3"].nunique(), "crops", world["panel"]["crop"].unique().tolist())
print(world["panel"].groupby("crop")["yield_tha"].median())
print("anomaly finite", world["panel"]["yield_anomaly"].notna().mean())
print(world["panel"][["iso3", "crop", "year", "yield_tha", "yield_anomaly"]].dropna().head())
