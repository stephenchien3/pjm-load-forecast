"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = REPO_ROOT / "data" / "sample.csv"


@pytest.fixture
def sample_csv_path() -> Path:
    """Path to the bundled sample PJM CSV."""
    return SAMPLE_CSV


@pytest.fixture
def synthetic_hourly_series() -> pd.DataFrame:
    """Small deterministic hourly series for unit tests (240 hours)."""
    idx = pd.date_range("2020-01-01", periods=240, freq="h")
    values = 1000 + 100 * np.sin(np.arange(240) / 24 * 2 * np.pi)
    return pd.DataFrame({"MW": values}, index=idx)
