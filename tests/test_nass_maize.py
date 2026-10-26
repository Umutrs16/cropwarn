import json
from pathlib import Path

import pandas as pd

from cropwarn.data.nass_maize import (
    BU_ACRE_TO_T_HA,
    MAIZE_CALENDAR,
    _canonical_yield_series,
    parse_state_corn_json,
    rollup_to_state,
    state_unit_id,
)


def test_maize_calendar_is_us_summer():
    sos, eos = MAIZE_CALENDAR
    assert sos == 120
    assert eos == 280
    assert eos - 192 < sos


def test_bushel_conversion_is_standard_corn():
    assert 0.062 < BU_ACRE_TO_T_HA < 0.064
    assert abs(160 * BU_ACRE_TO_T_HA - 10.04) < 0.05


def test_canonical_yield_picks_grain_not_silage():
    county = {
        "name": "MCLEAN",
        "series": [
            {
                "canonical": False,
                "statistic": "YIELD",
                "util_practice": "GRAIN",
                "unit": "BU / ACRE",
                "values": {"2012": 99.0},
            },
            {
                "canonical": True,
                "statistic": "YIELD",
                "util_practice": "SILAGE",
                "unit": "TONS / ACRE",
                "values": {"2012": 12.0},
            },
            {
                "canonical": True,
                "statistic": "YIELD",
                "util_practice": "GRAIN",
                "unit": "BU / ACRE",
                "values": {"2011": 175.0, "2012": 105.0, "2013": 178.0},
            },
        ],
    }
    series = _canonical_yield_series(county)
    assert series[2012] == 105.0
    assert 2011 in series


def test_parse_state_json_fixture(tmp_path: Path):
    payload = {
        "commodity": {"desc": "CORN", "slug": "corn"},
        "counties": {
            "113": {
                "name": "MCLEAN",
                "series": [
                    {
                        "canonical": True,
                        "statistic": "YIELD",
                        "util_practice": "GRAIN",
                        "unit": "BU / ACRE",
                        "values": {str(y): 170.0 if y != 2012 else 105.0 for y in range(2001, 2025)},
                    }
                ],
            }
        },
    }
    path = tmp_path / "17_corn.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    out = parse_state_corn_json(path, "17")
    assert out["unit_id"].eq("USA-IL-113").all()
    assert out["crop"].eq("maize").all()
    row = out[out["year"] == 2012].iloc[0]
    assert abs(row["yield_tha"] - 105.0 * BU_ACRE_TO_T_HA) < 1e-6
    assert row["yield_tha"] < 8.0


def test_state_unit_id_strips_county():
    assert state_unit_id("USA-IL-113") == "USA-IL"
    assert state_unit_id("USA-IA") == "USA-IA"


def test_rollup_averages_counties():
    frame = pd.DataFrame(
        {
            "iso3": ["USA", "USA"],
            "unit_id": ["USA-IL-001", "USA-IL-003"],
            "crop": ["maize", "maize"],
            "date": ["2012-07-15", "2012-07-15"],
            "evi": [0.2, 0.4],
            "sif": [0.3, 0.5],
        }
    )
    frame["date"] = pd.to_datetime(frame["date"])
    out = rollup_to_state(frame)
    assert len(out) == 1
    assert out.iloc[0]["unit_id"] == "USA-IL"
    assert abs(out.iloc[0]["evi"] - 0.3) < 1e-9
