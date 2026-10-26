from pathlib import Path

import pandas as pd

from cropwarn.data.enso import parse_oni
from cropwarn.revision.world_ops import attach_tropomi_as_sif, filter_world_years
from cropwarn.validation.bootstrap import cluster_bootstrap_lead


def test_parse_oni_uses_anomaly_column(tmp_path: Path):
    p = tmp_path / "oni.txt"
    p.write_text("SEAS  YR   TOTAL   ANOM\nOND  2015   26.1    2.30\nDJF  2016   27.0    2.20\n", encoding="utf-8")
    out = parse_oni(p)
    assert float(out.loc[out["season"] == "OND", "oni"].iloc[0]) == 2.30


def test_filter_world_years_keeps_trade():
    world = {
        "panel": pd.DataFrame({"year": [2014, 2016], "crop": ["wheat", "wheat"]}),
        "series": pd.DataFrame({"year": [2014, 2016], "sif": [1.0, 2.0]}),
        "trade": pd.DataFrame({"year": [2015]}),
    }
    out = filter_world_years(world, 2015)
    assert set(out["panel"]["year"]) == {2016}
    assert len(out["trade"]) == 1


def test_attach_tropomi_overwrites_sif_without_fill():
    world = {
        "series": pd.DataFrame(
            {
                "iso3": ["USA", "USA"],
                "crop": ["wheat", "wheat"],
                "date": pd.to_datetime(["2019-06-10", "2019-07-12"]),
                "sif": [9.0, 9.0],
            }
        )
    }
    tropomi = pd.DataFrame(
        {
            "iso3": ["USA"],
            "crop": ["wheat"],
            "date": pd.to_datetime(["2019-06-15"]),
            "tropomi_sif": [0.4],
        }
    )
    out = attach_tropomi_as_sif(world, tropomi)
    assert abs(out["series"].iloc[0]["sif"] - 0.4) < 1e-9


def test_cluster_bootstrap_returns_delta_interval():
    rows = []
    for year in range(2016, 2024):
        for crop in ("wheat", "soybean"):
            for lead in (0, 192):
                rows.append(
                    {
                        "experiment": "B2_optical_climate",
                        "iso3": "USA",
                        "crop": crop,
                        "year": year,
                        "lead": lead,
                        "yield_anomaly": 0.2 * (year - 2020) + (0.1 if crop == "wheat" else -0.1),
                        "pred": 0.15 * (year - 2020),
                    }
                )
                rows.append(
                    {
                        "experiment": "B4_plus_sif",
                        "iso3": "USA",
                        "crop": crop,
                        "year": year,
                        "lead": lead,
                        "yield_anomaly": 0.2 * (year - 2020) + (0.1 if crop == "wheat" else -0.1),
                        "pred": 0.16 * (year - 2020),
                    }
                )
    out = cluster_bootstrap_lead(pd.DataFrame(rows), n_boot=20, seed=0)
    assert "B4_minus_B2" in set(out["experiment"])
    assert "disolated_ge14" in out.columns