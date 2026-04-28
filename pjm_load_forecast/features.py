"""Feature engineering for PJM hourly load forecasting."""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


_HOLIDAY_CACHE: dict[tuple[pd.Timestamp, pd.Timestamp], pd.DatetimeIndex] = {}


def _us_holidays(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    key = (start.normalize(), end.normalize())
    if key not in _HOLIDAY_CACHE:
        cal = USFederalHolidayCalendar()
        _HOLIDAY_CACHE[key] = cal.holidays(start=key[0], end=key[1])
    return _HOLIDAY_CACHE[key]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with calendar features appended.

    Adds: ``hour``, ``dayofweek``, ``month``, ``is_weekend``,
    ``is_holiday`` (US federal holidays).
    """
    out = df.copy()
    idx = out.index
    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["month"] = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    holidays = _us_holidays(idx.min(), idx.max())
    out["is_holiday"] = idx.normalize().isin(holidays).astype(int)
    return out


def add_lag_features(*args, **kwargs):  # implemented in Task 6
    raise NotImplementedError


def add_rolling_features(*args, **kwargs):  # implemented in Task 7
    raise NotImplementedError


def build_design_matrix(*args, **kwargs):  # implemented in Task 7
    raise NotImplementedError
