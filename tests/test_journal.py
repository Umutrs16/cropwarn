import json
from pathlib import Path

from cropwarn.validation.go_nogo import supported_journal, write_report
from cropwarn.validation.journal import recommend_journal


def test_trade_without_phys_maps_to_nc():
    gates = {
        "dynamic_phenology": {"go": False, "value": -0.03, "threshold": 0.03},
        "physiology_lead": {"go": False, "value": 0.0, "threshold": 14},
        "uncertainty": {"go": True, "value": 0.01, "threshold": 0.08},
        "trade_value": {"go": True, "value": 192.0, "threshold": 16},
        "overall_go": False,
    }
    inventory = {
        "gosif": True,
        "bilateral_trade": True,
        "modis_optical": True,
        "headline_crops": ["wheat", "soybean"],
        "seasonal_rice_held_out": True,
        "usa_maize_held_out": True,
        "eu_maize_held_out": True,
    }
    verdict = recommend_journal(gates, inventory)
    assert verdict["recommendation"] == "Nature Communications / RSE"
    assert any("US NASS maize" in n for n in verdict["notes"])
    assert any("Eurostat NUTS maize" in n for n in verdict["notes"])
    assert supported_journal(gates) == "Nature Communications / RSE"


def test_write_report_names_nc_not_nature_food_overall(tmp_path: Path):
    report = {
        "physiology_lead": {"go": False, "value": 0.0, "threshold": 14, "fallback": "x"},
        "trade_value": {"go": True, "value": 192.0, "threshold": 16, "fallback": "x"},
        "dynamic_phenology": {"go": False, "value": -0.03, "threshold": 0.03, "fallback": "x"},
        "uncertainty": {"go": True, "value": 0.01, "threshold": 0.08, "fallback": "x"},
        "overall_go": False,
    }
    write_report(report, tmp_path / "go_nogo.json")
    md = (tmp_path / "go_nogo.md").read_text(encoding="utf-8")
    assert "Nature Food overall:** NO-GO" in md
    assert "Supported journal:** Nature Communications / RSE" in md
    payload = json.loads((tmp_path / "go_nogo.json").read_text(encoding="utf-8"))
    assert payload["overall_go"] is False
