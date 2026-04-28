"""CSV loading and splitting for PJM hourly load data."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_pjm_csv(path: str | Path) -> pd.DataFrame:
    """Load a PJM hourly load CSV.

    The file is expected to have two columns: ``Datetime`` (parseable
    timestamp) and ``MW`` (numeric load). Duplicate timestamps — which
    PJM emits at DST fall-back hours — are averaged.

    Returns a DataFrame indexed by sorted ``DatetimeIndex`` with a single
    float column ``MW``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, parse_dates=["Datetime"])
    df = df.rename(columns=str.strip)
    df = df.set_index("Datetime").sort_index()
    df["MW"] = df["MW"].astype(float)
    df = df.groupby(level=0).mean()
    return df[["MW"]]


def split_by_date(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-indexed DataFrame chronologically.

    The remaining fraction (``1 - train_frac - val_frac``) goes to the
    test set. Splits are by row position on the sorted index, so each
    split is a contiguous time window with no overlap.
    """
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError(
            "train_frac and val_frac must be positive and sum to less than 1; "
            f"got train_frac={train_frac}, val_frac={val_frac}"
        )
    df = df.sort_index()
    n = len(df)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    return train, val, test
