"""Forecasting models for PJM hourly load.

All models implement ``fit(X, y)`` and ``predict(X) -> np.ndarray`` so
they're interchangeable in the backtest harness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class SeasonalNaive:
    """Predicts ``y_t = y_{t-lag}`` by reading the ``lag_{lag}`` column.

    A trivial baseline; ``fit`` is a no-op so the interface matches the
    other models.
    """

    lag: int = 168

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalNaive":  # noqa: ARG002
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        col = f"lag_{self.lag}"
        if col not in X.columns:
            raise KeyError(
                f"SeasonalNaive(lag={self.lag}) needs column {col!r} in X"
            )
        return X[col].to_numpy(dtype=float)


class LinearModel:  # implemented in next task
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class GradientBoostingModel:  # implemented in next task
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
