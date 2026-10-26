import json
from pathlib import Path

from cropwarn.data.eurostat_maize import (
    EU_MAIZE_CALENDAR,
    _is_region,
    parse_jsonstat_geo_time,
)


def test_eu_maize_calendar_is_summer():
    sos, eos = EU_MAIZE_CALENDAR
    assert sos == 120
    assert 260 <= eos <= 290
    assert eos - 192 < sos


def test_nuts2_filter():
    assert _is_region("FRK2")
    assert _is_region("DE2")
    assert not _is_region("DE21")
    assert not _is_region("FR")
    assert not _is_region("FRK")
    assert not _is_region("FRY1")
    assert not _is_region("US01")


def test_jsonstat_geo_time(tmp_path: Path):
    payload = {
        "value": {"7": 8.5, "32": 10.2},
        "id": ["freq", "crops", "strucpro", "geo", "time"],
        "dimension": {
            "geo": {"category": {"index": {"DE21": 0, "FRK2": 1}, "label": {"DE21": "Oberbayern", "FRK2": "Rhone"}}},
            "time": {"category": {"index": {str(y): i for i, y in enumerate(range(2000, 2025))}, "label": {}}},
        },
    }
    path = tmp_path / "t.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    out = parse_jsonstat_geo_time(path, "yield_tha")
    row = out[(out["geo"] == "DE21") & (out["year"] == 2007)].iloc[0]
    assert abs(row["yield_tha"] - 8.5) < 1e-6
    row2 = out[(out["geo"] == "FRK2") & (out["year"] == 2007)].iloc[0]
    assert abs(row2["yield_tha"] - 10.2) < 1e-6
