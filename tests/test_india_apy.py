import pandas as pd

from cropwarn.data.india_apy import MAIN_MONSOON_SEASON, _name_key, _year_from_apy, parse_india_kharif_rice
from cropwarn.data.labels import rolling_trend_anomaly


def test_apy_year_maps_agricultural_year():
    assert _year_from_apy("2019-20") == 2019
    assert _year_from_apy(2016) == 2016


def test_district_name_key_is_order_invariant():
    assert _name_key("24 Parganas South") == _name_key("South 24 Parganas")


def test_east_india_uses_aman_winter_not_kharif():
    assert MAIN_MONSOON_SEASON["west bengal"] == "winter"
    assert MAIN_MONSOON_SEASON["odisha"] == "winter"
    assert "punjab" not in MAIN_MONSOON_SEASON


def test_unit_id_detrend_is_independent():
    rows = []
    for year in range(2001, 2011):
        rows.append({"iso3": "IND", "unit_id": "IND-PB", "crop": "rice", "year": year, "yield_tha": 4.0 + 0.05 * (year - 2001)})
        rows.append({"iso3": "IND", "unit_id": "IND-WB", "crop": "rice", "year": year, "yield_tha": 2.0 + (1.5 if year == 2010 else 0)})
    out = rolling_trend_anomaly(pd.DataFrame(rows), min_history=5)
    pb = out[(out["unit_id"] == "IND-PB") & (out["year"] == 2010)].iloc[0]
    wb = out[(out["unit_id"] == "IND-WB") & (out["year"] == 2010)].iloc[0]
    assert abs(pb["yield_anomaly"]) < 0.2
    assert wb["yield_anomaly"] > 1.0


def test_parse_india_kharif_rice_if_present():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    out = parse_india_kharif_rice(root)
    if out.empty:
        return
    assert set(out["crop"]) == {"rice"}
    assert out["iso3"].eq("IND").all()
    assert out["unit_id"].str.startswith("IND-").all()
    assert out["yield_tha"].between(0.5, 7).all()
    assert out.groupby("unit_id")["year"].nunique().min() >= 12
