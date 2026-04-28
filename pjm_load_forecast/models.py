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


class LinearModel:
    """Ridge regression on the full design matrix, with feature scaling."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._pipeline: Optional[Pipeline] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearModel":
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )
        self._pipeline.fit(X.to_numpy(), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("LinearModel.predict() called before fit()")
        return self._pipeline.predict(X.to_numpy())


class GradientBoostingModel:
    """Histogram gradient boosting on the full design matrix."""

    def __init__(
        self,
        max_iter: int = 200,
        learning_rate: float = 0.05,
        max_depth: Optional[int] = 8,
        random_state: int = 0,
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self._model: Optional[HistGradientBoostingRegressor] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoostingModel":
        self._model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self._model.fit(X.to_numpy(), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("GradientBoostingModel.predict() called before fit()")
        return self._model.predict(X.to_numpy())
