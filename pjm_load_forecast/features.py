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


def add_lag_features(
    df: pd.DataFrame,
    lags: Iterable[int],
    column: str = "MW",
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``lag_{k}`` columns for each ``k`` in ``lags``."""
    lags = list(lags)
    if any(l <= 0 for l in lags):
        raise ValueError(f"All lags must be positive; got {lags}")
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[column].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    windows: Iterable[int],
    column: str = "MW",
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``rollmean_{w}`` columns for each window.

    Each rolling mean uses only past values: it is computed on a 1-step
    shifted series, so no future leakage.
    """
    windows = list(windows)
    if any(w <= 0 for w in windows):
        raise ValueError(f"All windows must be positive; got {windows}")
    out = df.copy()
    shifted = out[column].shift(1)
    for w in windows:
        out[f"rollmean_{w}"] = shifted.rolling(window=w, min_periods=w).mean()
    return out


def build_design_matrix(
    df: pd.DataFrame,
    target: str = "MW",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Drop rows with NaNs and return ``(X, y)`` aligned by index.

    The target column is removed from ``X``.
    """
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in DataFrame")
    clean = df.dropna()
    y = clean[target]
    X = clean.drop(columns=[target])
    return X, y
