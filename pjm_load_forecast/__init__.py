"""PJM hourly load forecasting.

A small package that loads a PJM hourly-load CSV, builds calendar / lag
features, and forecasts the next hour with two models:

- ``SeasonalNaive`` — predicts the value 168 hours (one week) ago. A
  surprisingly hard baseline to beat for hourly load.
- ``GradientBoostingModel`` — scikit-learn ``HistGradientBoostingRegressor``
  on calendar + lag features.

Use :func:`walk_forward_backtest` to compare them on a held-out tail of
the series.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

__version__ = "0.1.0"

__all__ = [
    "load_pjm_csv",
    "build_features",
    "SeasonalNaive",
    "GradientBoostingModel",
    "mae",
    "rmse",
    "mape",
    "walk_forward_backtest",
]


# ---------- data loading ----------

def load_pjm_csv(path: str | Path) -> pd.DataFrame:
    """Load a PJM hourly load CSV into a sorted, deduplicated DataFrame.

    The file must have a ``Datetime`` column and a load column named either
    ``MW`` or zone-prefixed (``PJME_MW``, ``AEP_MW``, ...). Duplicate
    timestamps — which PJM emits at DST fall-back hours — are averaged.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["Datetime"])
    mw_cols = [c for c in df.columns if c.upper().endswith("MW")]
    if not mw_cols:
        raise ValueError(f"No MW column in {path}; columns: {list(df.columns)}")
    df = df.rename(columns={mw_cols[0]: "MW"})
    df = df.set_index("Datetime").sort_index()
    df["MW"] = df["MW"].astype(float)
    return df.groupby(level=0).mean()[["MW"]]


# ---------- feature engineering ----------

def build_features(df: pd.DataFrame, lags=(1, 24, 168)) -> pd.DataFrame:
    """Append calendar features and past-value lags to ``df``.

    Adds: ``hour``, ``dayofweek``, ``month``, ``is_weekend``, and one
    ``lag_{k}`` column per element of ``lags``. Returns a copy with the
    leading rows dropped (they have NaN lags).
    """
    if any(l <= 0 for l in lags):
        raise ValueError(f"lags must all be positive; got {lags}")
    out = df.copy()
    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    out["month"] = out.index.month
    out["is_weekend"] = (out.index.dayofweek >= 5).astype(int)
    for lag in lags:
        out[f"lag_{lag}"] = out["MW"].shift(lag)
    return out.dropna()


# ---------- models ----------

class SeasonalNaive:
    """Predict ``y_t = y_{t-lag}`` by reading the ``lag_{lag}`` column."""

    def __init__(self, lag: int = 168):
        self.lag = lag

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalNaive":  # noqa: ARG002
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        col = f"lag_{self.lag}"
        if col not in X.columns:
            raise KeyError(f"SeasonalNaive needs column {col!r}")
        return X[col].to_numpy(dtype=float)


class GradientBoostingModel:
    """``HistGradientBoostingRegressor`` wrapped in a fit/predict interface."""

    def __init__(self, max_iter: int = 200, learning_rate: float = 0.05,
                 max_depth: Optional[int] = 8, random_state: int = 0):
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
            raise RuntimeError("predict() called before fit()")
        return self._model.predict(X.to_numpy())


# ---------- metrics ----------

def _check(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    _check(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    _check(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, ignoring rows where ``y_true == 0``."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    _check(y_true, y_pred)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


# ---------- backtest ----------

def walk_forward_backtest(model, df: pd.DataFrame, target: str = "MW",
                          train_size: int = 24 * 30, horizon: int = 24,
                          step: int = 168) -> dict:
    """Expanding-window walk-forward backtest.

    Refit ``model`` every ``step`` rows on all data so far, predict the
    next ``horizon`` rows, slide forward, repeat. Returns a dict with
    ``mae``, ``rmse``, ``mape``, ``n_windows``, and a stitched
    ``predictions`` Series.
    """
    if train_size <= 0 or horizon <= 0 or step <= 0:
        raise ValueError("train_size, horizon, step must be positive")
    df = df.sort_index()
    feats = [c for c in df.columns if c != target]
    preds, truths = [], []
    start, n = train_size, len(df)
    while start + horizon <= n:
        train, test = df.iloc[:start], df.iloc[start:start + horizon]
        model.fit(train[feats], train[target])
        preds.append(pd.Series(model.predict(test[feats]), index=test.index))
        truths.append(test[target])
        start += step
    if not preds:
        raise ValueError("no backtest windows produced — dataset too short")
    p, t = pd.concat(preds), pd.concat(truths)
    return {
        "mae": mae(t.to_numpy(), p.to_numpy()),
        "rmse": rmse(t.to_numpy(), p.to_numpy()),
        "mape": mape(t.to_numpy(), p.to_numpy()),
        "n_windows": len(preds),
        "predictions": p,
    }
