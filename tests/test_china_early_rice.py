import pandas as pd

from cropwarn.data.china_early_rice import (
    EARLY_RICE_CALENDAR,
    _code_for,
    _parse_bulletin_html,
    parse_china_early_rice,
)


def test_early_rice_calendar_is_pre_monsoon_harvest():
    sos, eos = EARLY_RICE_CALENDAR
    assert sos == 90
    assert 180 <= eos <= 210


def test_province_codes_from_en_and_zh():
    assert _code_for("Zhejiang") == "ZJ"
    assert _code_for("浙 江") == "ZJ"
    assert _code_for("Guangxi Zhuang") == "GX"
    assert _code_for("Hunan") == "HN"
    assert _code_for("Anhui Province") == "AH"
    assert _code_for("Guangzhou Province") == "GD"
    assert _code_for("Hainan Province") == "HI"
    assert _code_for("Guangxi Zhuang Autonomous Region") == "GX"
    assert _code_for("National Total") is None
    assert _code_for("Heilongjiang") is None


def test_parse_english_bulletin_table():
    html = """
    <table>
      <tr><th>Regions</th><th>Sown Area (thousand hectares)</th>
          <th>Total Outputs (10,000 tons)</th><th>Yield per Unit Area (kg)</th></tr>
      <tr><td>National Total</td><td>4755.1</td><td>2812.3</td><td>5914.3</td></tr>
      <tr><td>Zhejiang</td><td>113.1</td><td>70.2</td><td>6206.5</td></tr>
      <tr><td>Jiangxi</td><td>1220.1</td><td>677.2</td><td>5550.2</td></tr>
      <tr><td>Heilongjiang</td><td></td><td></td><td></td></tr>
    </table>
    """
    out = _parse_bulletin_html(html, 2022)
    assert set(out["state_code"]) == {"ZJ", "JX"}
    zj = out[out["state_code"] == "ZJ"].iloc[0]
    assert abs(zj["yield_tha"] - 6.2065) < 1e-3
    assert abs(zj["harvested_ha"] - 113100) < 1


def test_parse_china_early_rice_if_present():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cache = root / "data_interim" / "china_early_rice.csv"
    if not cache.exists():
        return
    out = parse_china_early_rice(root)
    if out.empty:
        return
    assert set(out["crop"]) == {"rice"}
    assert out["iso3"].eq("CHN").all()
    assert out["unit_id"].str.startswith("CHN-").all()
    assert out["yield_tha"].between(3.5, 8.5).all()
    assert out.groupby("unit_id")["year"].nunique().min() >= 6
    ah = out[(out["state_code"] == "AH") & (out["year"] == 2020)]
    if len(ah):
        assert ah.iloc[0]["yield_tha"] < 5.5
