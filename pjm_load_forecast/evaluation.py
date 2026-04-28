"""Evaluation metrics and walk-forward backtesting."""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd


def _check_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _check_shapes(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _check_shapes(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, skipping rows where y_true == 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _check_shapes(y_true, y_pred)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def walk_forward_backtest(
    model,
    df: pd.DataFrame,
    target: str = "MW",
    train_size: int = 24 * 30,
    horizon: int = 24,
    step: int = 168,
) -> dict:
    """Expanding-window walk-forward backtest.

    Starts with the first ``train_size`` rows as training data, fits the
    model, predicts the next ``horizon`` rows, slides the train window
    forward by ``step`` rows, refits, and repeats until the data runs
    out. Returns aggregated metrics over all out-of-sample predictions
    and a Series of stitched predictions indexed by timestamp.
    """
    if train_size <= 0 or horizon <= 0 or step <= 0:
        raise ValueError(
            f"train_size, horizon, step must all be positive; "
            f"got train_size={train_size}, horizon={horizon}, step={step}"
        )

    df = df.sort_index()
    feature_cols = [c for c in df.columns if c != target]

    preds_pieces: list[pd.Series] = []
    truths_pieces: list[pd.Series] = []
    n_windows = 0

    start = train_size
    n = len(df)
    while start + horizon <= n:
        train = df.iloc[:start]
        test = df.iloc[start : start + horizon]

        X_train = train[feature_cols]
        y_train = train[target]
        X_test = test[feature_cols]
        y_test = test[target]

        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)

        preds_pieces.append(pd.Series(y_hat, index=test.index))
        truths_pieces.append(y_test)
        n_windows += 1
        start += step

    if n_windows == 0:
        raise ValueError(
            "No backtest windows produced; train_size + horizon exceeds dataset length"
        )

    preds = pd.concat(preds_pieces)
    truths = pd.concat(truths_pieces)

    return {
        "mae": mae(truths.to_numpy(), preds.to_numpy()),
        "rmse": rmse(truths.to_numpy(), preds.to_numpy()),
        "mape": mape(truths.to_numpy(), preds.to_numpy()),
        "n_windows": n_windows,
        "predictions": preds,
    }
