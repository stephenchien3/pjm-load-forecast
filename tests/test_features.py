"""Tests for pjm_load_forecast.features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pjm_load_forecast.features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    build_design_matrix,
)


class TestCalendarFeatures:
    def test_columns_added(self, synthetic_hourly_series):
        out = add_calendar_features(synthetic_hourly_series)
        for col in ["hour", "dayofweek", "month", "is_weekend", "is_holiday"]:
            assert col in out.columns

    def test_known_values(self):
        # 2020-01-01 was a Wednesday and a US federal holiday
        idx = pd.date_range("2020-01-01 00:00:00", periods=3, freq="h")
        df = pd.DataFrame({"MW": [1.0, 2.0, 3.0]}, index=idx)
        out = add_calendar_features(df)
        assert out["hour"].tolist() == [0, 1, 2]
        assert out["dayofweek"].tolist() == [2, 2, 2]
        assert out["month"].tolist() == [1, 1, 1]
        assert out["is_weekend"].tolist() == [0, 0, 0]
        assert out["is_holiday"].tolist() == [1, 1, 1]

    def test_weekend_flag(self):
        # 2020-01-04 was a Saturday
        idx = pd.date_range("2020-01-04 12:00:00", periods=2, freq="D")
        df = pd.DataFrame({"MW": [1.0, 2.0]}, index=idx)
        out = add_calendar_features(df)
        assert out["is_weekend"].tolist() == [1, 1]

    def test_does_not_mutate_input(self, synthetic_hourly_series):
        before = synthetic_hourly_series.copy()
        _ = add_calendar_features(synthetic_hourly_series)
        pd.testing.assert_frame_equal(synthetic_hourly_series, before)
