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


class TestLagFeatures:
    def test_columns_added(self, synthetic_hourly_series):
        out = add_lag_features(synthetic_hourly_series, lags=[1, 24, 168])
        assert "lag_1" in out.columns
        assert "lag_24" in out.columns
        assert "lag_168" in out.columns

    def test_lag_equals_shifted_value(self, synthetic_hourly_series):
        out = add_lag_features(synthetic_hourly_series, lags=[1, 5])
        for lag in [1, 5]:
            expected = synthetic_hourly_series["MW"].shift(lag)
            pd.testing.assert_series_equal(
                out[f"lag_{lag}"], expected, check_names=False
            )

    def test_empty_lags_is_noop(self, synthetic_hourly_series):
        out = add_lag_features(synthetic_hourly_series, lags=[])
        pd.testing.assert_frame_equal(out, synthetic_hourly_series)

    def test_negative_lag_raises(self, synthetic_hourly_series):
        with pytest.raises(ValueError):
            add_lag_features(synthetic_hourly_series, lags=[-1])


class TestRollingFeatures:
    def test_columns_added(self, synthetic_hourly_series):
        out = add_rolling_features(synthetic_hourly_series, windows=[24, 168])
        assert "rollmean_24" in out.columns
        assert "rollmean_168" in out.columns

    def test_rolling_mean_uses_only_past(self, synthetic_hourly_series):
        out = add_rolling_features(synthetic_hourly_series, windows=[3])
        mw = synthetic_hourly_series["MW"]
        expected = mw.shift(1).rolling(window=3, min_periods=3).mean()
        pd.testing.assert_series_equal(
            out["rollmean_3"], expected, check_names=False
        )

    def test_invalid_window_raises(self, synthetic_hourly_series):
        with pytest.raises(ValueError):
            add_rolling_features(synthetic_hourly_series, windows=[0])


class TestBuildDesignMatrix:
    def test_returns_aligned_x_y_with_no_nans(self, synthetic_hourly_series):
        feats = add_calendar_features(synthetic_hourly_series)
        feats = add_lag_features(feats, lags=[1, 24])
        feats = add_rolling_features(feats, windows=[24])
        X, y = build_design_matrix(feats, target="MW")
        assert not X.isna().any().any()
        assert not y.isna().any()
        assert len(X) == len(y)
        assert "MW" not in X.columns
        assert len(X) == len(synthetic_hourly_series) - 24

    def test_target_must_exist(self, synthetic_hourly_series):
        with pytest.raises(KeyError):
            build_design_matrix(synthetic_hourly_series, target="nope")
