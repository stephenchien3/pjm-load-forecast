"""Tests for pjm_load_forecast.evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pjm_load_forecast.evaluation import (
    mae,
    mape,
    rmse,
    walk_forward_backtest,
)
from pjm_load_forecast.models import SeasonalNaive


class TestMetrics:
    def test_mae(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.0, 4.0])
        assert mae(y_true, y_pred) == pytest.approx(0.5)

    def test_rmse(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        assert rmse(y_true, y_pred) == pytest.approx(1.0)

    def test_mape(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        assert mape(y_true, y_pred) == pytest.approx(0.1)

    def test_mape_skips_zero_truth(self):
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([10.0, 110.0])
        assert mape(y_true, y_pred) == pytest.approx(0.1)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            mae(np.array([1.0, 2.0]), np.array([1.0]))
        with pytest.raises(ValueError):
            rmse(np.array([1.0, 2.0]), np.array([1.0]))
        with pytest.raises(ValueError):
            mape(np.array([1.0, 2.0]), np.array([1.0]))
