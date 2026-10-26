import numpy as np
import pandas as pd

from cropwarn.phenology.detect import estimate_sos_eos, realtime_sos


def test_evi_sos_eos_follows_calendar_window_not_offseason_peak():
    dates = pd.date_range("2012-01-01", "2012-12-31", freq="16D")
    doy = dates.dayofyear.to_numpy()
    # Crop greenup around DOY 120, senescence around 270; a fake winter peak at DOY 20.
    evi = 0.15 + 0.45 * np.exp(-0.5 * ((doy - 190) / 40) ** 2)
    evi = np.where(doy < 40, 0.55, evi)
    sos, eos = estimate_sos_eos(dates, evi, sos_prior=120, eos_prior=270, search_days=50)
    assert 90 <= sos <= 170
    assert 220 <= eos <= 310


def test_realtime_sos_falls_back_before_greenup():
    dates = pd.date_range("2012-04-01", "2012-10-01", freq="16D")
    evi = np.linspace(0.1, 0.6, len(dates))
    sos = realtime_sos(dates, evi, sos_prior=120, eos_prior=270, cutoff=pd.Timestamp("2012-03-01"))
    assert sos == 120
