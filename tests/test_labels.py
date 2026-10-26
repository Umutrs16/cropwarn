import pandas as pd

from cropwarn.data.labels import rolling_trend_anomaly


def test_rolling_trend_uses_only_past_years():
    rows = []
    for year, yld in [(2001, 2.0), (2002, 2.2), (2003, 2.4), (2004, 2.6), (2005, 2.8), (2006, 10.0)]:
        rows.append({"iso3": "AAA", "crop": "wheat", "year": year, "yield_tha": yld})
    out = rolling_trend_anomaly(pd.DataFrame(rows), min_history=5)
    row = out[out["year"] == 2006].iloc[0]
    # Trend fitted on 2.0...2.8 must not jump toward the 10 t/ha test year.
    assert row["yield_trend"] < 4.0
    assert row["yield_anomaly"] > 5.0


def test_first_difference_and_theil_sen_are_causal():
    rows = []
    for year, yld in [(2001, 2.0), (2002, 2.1), (2003, 2.2), (2004, 2.3), (2005, 2.4), (2006, 9.0)]:
        rows.append({"iso3": "AAA", "crop": "wheat", "year": year, "yield_tha": yld})
    panel = pd.DataFrame(rows)
    fd = rolling_trend_anomaly(panel, method="first_difference", min_history=4)
    row = fd[fd["year"] == 2006].iloc[0]
    assert abs(row["yield_anomaly"] - (9.0 - 2.4)) < 1e-9
    ts = rolling_trend_anomaly(panel, method="theil_sen", min_history=4)
    ts_row = ts[ts["year"] == 2006].iloc[0]
    assert ts_row["yield_trend"] < 4.0
    pct = rolling_trend_anomaly(panel, method="percent", min_history=4)
    assert abs(pct[pct["year"] == 2006].iloc[0]["yield_anomaly"]) > 1.0
