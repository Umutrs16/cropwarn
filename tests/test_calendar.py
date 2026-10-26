from cropwarn.constants import RICE_DUAL_SEASONS, STATIC_CALENDAR
from cropwarn.data.observational import _calendar


def test_vietnam_rice_uses_winter_spring_not_summer():
    sos, eos = _calendar("rice", "VNM")
    assert sos >= 300
    assert eos <= 150


def test_china_rice_window_is_middle_late_not_both_crops():
    sos, eos = STATIC_CALENDAR["rice"]["CHN"]
    assert sos >= 130
    assert eos - sos <= 150


def test_dual_rice_covers_the_double_crop_countries():
    for iso3 in ("CHN", "VNM", "BGD", "PHL", "IND"):
        assert iso3 in RICE_DUAL_SEASONS
        assert RICE_DUAL_SEASONS[iso3] != _calendar("rice", iso3)
