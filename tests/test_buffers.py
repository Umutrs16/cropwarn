import pandas as pd

from cropwarn.trade.buffers import importer_accounts, skill_score


def test_idr_is_high_for_import_dependent_and_zero_for_net_exporter():
    trade = pd.DataFrame(
        [
            {"year": 2010, "crop": "wheat", "exporter": "USA", "importer": "EGY", "qty": 80.0},
            {"year": 2010, "crop": "wheat", "exporter": "USA", "importer": "USA", "qty": 1.0},
        ]
    )
    labels = pd.DataFrame(
        [
            {"year": 2010, "crop": "wheat", "iso3": "EGY", "production_t": 20.0},
            {"year": 2010, "crop": "wheat", "iso3": "USA", "production_t": 200.0},
        ]
    )
    acc = importer_accounts(trade, labels)
    egy = acc[(acc["iso3"] == "EGY")].iloc[0]
    usa = acc[(acc["iso3"] == "USA")].iloc[0]
    assert egy["idr"] > 0.7
    assert usa["idr"] < 0.05


def test_skill_score_zero_at_prevalence():
    assert abs(skill_score(0.26, 0.26)) < 1e-9
    assert skill_score(1.0, 0.26) == 1.0
