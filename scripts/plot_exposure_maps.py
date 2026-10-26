"""Build Figure 4: observed vs 192-d predicted importer exposure maps."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.data.io import read_table, write_table
from cropwarn.trade.exposure import importer_exposure_rows
from cropwarn.viz.figures import fig4_exposure_maps

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures"


def main() -> None:
    preds = read_table(ROOT / "results" / "predictions.csv")
    labels = read_table(ROOT / "data_interim" / "obs_panel_labels.csv")
    trade = read_table(ROOT / "data_interim" / "obs_trade_network.csv")
    food = preds[preds["experiment"] == "P_food_system"].copy()
    expo = importer_exposure_rows(food, trade, labels)
    write_table(expo, ROOT / "results" / "revision_p2" / "importer_exposure_rows.csv")
    path = fig4_exposure_maps(expo, FIGS, years=(2018, 2020, 2022), lead=192)
    print(path, path.stat().st_size, "importers", expo["importer"].nunique())


if __name__ == "__main__":
    main()
