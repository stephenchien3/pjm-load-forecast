"""Tests for pjm_load_forecast.models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pjm_load_forecast.models import (
    GradientBoostingModel,
    LinearModel,
    SeasonalNaive,
)


def _design_matrix_from_series(series: pd.Series, lag: int) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.DataFrame({"MW": series})
    df[f"lag_{lag}"] = df["MW"].shift(lag)
    df = df.dropna()
    return df.drop(columns=["MW"]), df["MW"]


class TestSeasonalNaive:
    def test_predict_returns_lag_column(self):
        idx = pd.date_range("2020-01-01", periods=400, freq="h")
        rng = np.random.default_rng(0)
        y = pd.Series(rng.normal(1000, 50, 400), index=idx)
        X, y_aligned = _design_matrix_from_series(y, lag=168)
        model = SeasonalNaive(lag=168)
        model.fit(X, y_aligned)
        preds = model.predict(X)
        np.testing.assert_array_equal(preds, X["lag_168"].to_numpy())

    def test_predict_shape(self):
        X = pd.DataFrame({"lag_168": np.arange(10.0)})
        y = pd.Series(np.arange(10.0))
        model = SeasonalNaive()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (10,)

    def test_missing_lag_column_raises(self):
        X = pd.DataFrame({"hour": np.arange(5)})
        y = pd.Series(np.arange(5.0))
        model = SeasonalNaive(lag=168)
        with pytest.raises(KeyError):
            model.predict(X)
