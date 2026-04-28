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


def split_by_date(*args, **kwargs):  # implemented in Task 4
    raise NotImplementedError
