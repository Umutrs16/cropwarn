import pandas as pd

from cropwarn.phenology.alignment import PhenologyVersion, align_cutoff, season_progress_signed, stage_name


def test_realtime_phenology_does_not_use_future_eos_at_long_lead():
    history = pd.DataFrame(
        {
            "iso3": ["IND", "IND"],
            "crop": ["rice", "rice"],
            "year": [2010, 2011],
            "eos_true": [300, 305],
            "eos_static": [320, 320],
        }
    )
    row = pd.Series(
        {
            "iso3": "IND",
            "crop": "rice",
            "year": 2012,
            "sos_static": 170,
            "eos_static": 320,
            "sos_true": 200,
            "eos_true": 340,
        }
    )
    meta = align_cutoff(row, lead=128, history=history, version=PhenologyVersion.DYNAMIC)
    assert meta["eos_used"] != 340
    assert meta["progress_at_cutoff"] <= 100


def test_realtime_sos_does_not_use_unobserved_greenup():
    history = pd.DataFrame(
        {
            "iso3": ["IND"],
            "crop": ["rice"],
            "year": [2011],
            "sos_true": [175],
            "eos_true": [300],
            "eos_static": [320],
        }
    )
    row = pd.Series(
        {
            "iso3": "IND",
            "crop": "rice",
            "year": 2012,
            "sos_static": 170,
            "eos_static": 320,
            "sos_true": 250,
            "eos_true": 340,
        }
    )
    meta = align_cutoff(row, lead=128, history=history, version=PhenologyVersion.DYNAMIC)
    # Cutoff is ~DOY 212, before the unobserved SOS 250.
    assert meta["sos_used"] != 250


def test_signed_progress_is_negative_before_sowing():
    assert season_progress_signed(50, 120, 280) < 0
    assert stage_name(season_progress_signed(50, 120, 280)) == "pre_sowing"
    assert stage_name(60.0) == "flowering"
