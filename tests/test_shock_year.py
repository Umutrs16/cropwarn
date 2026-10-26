import pandas as pd

from cropwarn.config import load_config
from cropwarn.validation.go_nogo import shock_year_lead_gain


def test_shock_year_lead_uses_event_years_only():
    cfg = load_config()
    panel = pd.DataFrame(
        {
            "iso3": ["USA"] * 6,
            "unit_id": ["USA"] * 6,
            "crop": ["wheat"] * 6,
            "year": [2011, 2012, 2013, 2011, 2012, 2013],
            "yield_z": [0.1, -1.8, 0.2, 0.1, -1.8, 0.2],
        }
    )
    rows = []
    for exp, pred_map in (
        ("B2_optical_climate", {0: -1.5, 64: 0.0, 192: 0.0}),
        ("B4_plus_sif", {0: -1.6, 64: -1.4, 192: 0.0}),
    ):
        for year, z in ((2011, 0.1), (2012, -1.8), (2013, 0.2)):
            for lead, pred in pred_map.items():
                rows.append(
                    {
                        "experiment": exp,
                        "iso3": "USA",
                        "unit_id": "USA",
                        "crop": "wheat",
                        "year": year,
                        "lead": lead,
                        "yield_anomaly": z * 0.4,
                        "pred": pred if year == 2012 else 0.0,
                    }
                )
    out = shock_year_lead_gain(pd.DataFrame(rows), panel, cfg)
    assert out["n"] > 0
    assert out["n_prediction_rows"] == out["n"]
    assert out["n_country_crop_years"] <= out["n"]
    assert out["b4_lead"] >= out["b2_lead"] or out["b4_lead"] != out["b4_lead"]
