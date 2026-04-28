# PJM Hourly Load Forecasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an installable Python package `pjm_load_forecast` that loads PJM hourly load CSVs, engineers calendar/lag/rolling features, trains three forecasters (SeasonalNaive, Ridge, HistGradientBoosting), and evaluates them via walk-forward backtesting — with a CLI, README, demo notebook, and ≥80% test coverage.

**Architecture:** Five focused modules — `data`, `features`, `models`, `evaluation`, `cli` — each with one responsibility, tested in isolation. Models share a `fit/predict` interface so they're interchangeable in the backtest harness. Walk-forward backtest is the headline evaluation: refit weekly, predict 24h ahead.

**Tech Stack:** Python ≥ 3.10, numpy, pandas, scikit-learn, joblib. Dev: pytest, pytest-cov, ruff. No deep-learning deps.

**Spec:** `docs/superpowers/specs/2026-04-28-pjm-load-forecast-design.md`

---

## File Map

Files this plan creates (in order they appear in tasks):

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, deps, entry point |
| `pjm_load_forecast/__init__.py` | Public re-exports + version |
| `data/sample.csv` | ~500-row PJM-format fixture for tests + demo |
| `pjm_load_forecast/data.py` | CSV loading, dedup, split |
| `tests/test_data.py` | Unit tests for `data.py` |
| `pjm_load_forecast/features.py` | Calendar, lag, rolling, design-matrix builders |
| `tests/test_features.py` | Unit tests for `features.py` |
| `pjm_load_forecast/models.py` | SeasonalNaive, LinearModel, GradientBoostingModel |
| `tests/test_models.py` | Unit tests for `models.py` |
| `pjm_load_forecast/evaluation.py` | mape/rmse/mae + walk_forward_backtest |
| `tests/test_evaluation.py` | Unit tests for `evaluation.py` |
| `pjm_load_forecast/cli.py` | argparse entry point |
| `tests/test_cli.py` | Subprocess CLI smoke tests |
| `tests/conftest.py` | Shared fixtures (sample CSV path, synthetic series) |
| `README.md` | Purpose, dataset, install, usage |
| `Makefile` | install / lint / test / coverage targets |
| `notebooks/demo.ipynb` | End-to-end walkthrough |
| `scripts/download_sample.py` | Helper to fetch full PJM CSV |

---

## Task 1: Project scaffold (pyproject.toml + package init)

**Files:**
- Create: `pyproject.toml`
- Create: `pjm_load_forecast/__init__.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pjm_load_forecast"
version = "0.1.0"
description = "Hourly electricity load forecasting for the PJM grid."
readme = "README.md"
requires-python = ">=3.10"
authors = [{name = "Stephen Chien", email = "stephenlchien@gmail.com"}]
license = {text = "MIT"}
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "joblib>=1.3",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.4",
]

[project.scripts]
pjm-forecast = "pjm_load_forecast.cli:main"

[tool.setuptools.packages.find]
include = ["pjm_load_forecast*"]

[tool.setuptools.package-data]
pjm_load_forecast = []

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.coverage.run]
source = ["pjm_load_forecast"]
omit = ["*/cli.py"]

[tool.coverage.report]
exclude_lines = ["pragma: no cover", "if __name__ == .__main__.:"]
```

- [ ] **Step 2: Write `pjm_load_forecast/__init__.py`**

```python
"""PJM hourly load forecasting package."""

from pjm_load_forecast.data import load_pjm_csv, split_by_date
from pjm_load_forecast.features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    build_design_matrix,
)
from pjm_load_forecast.models import (
    GradientBoostingModel,
    LinearModel,
    SeasonalNaive,
)
from pjm_load_forecast.evaluation import mae, mape, rmse, walk_forward_backtest

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "load_pjm_csv",
    "split_by_date",
    "add_calendar_features",
    "add_lag_features",
    "add_rolling_features",
    "build_design_matrix",
    "SeasonalNaive",
    "LinearModel",
    "GradientBoostingModel",
    "mae",
    "mape",
    "rmse",
    "walk_forward_backtest",
]
```

Note: this `__init__.py` references modules that don't exist yet. That's fine — it will fail to import until those modules exist, which is what makes later test failures meaningful.

- [ ] **Step 3: Create empty module placeholders so imports don't fail**

Create empty files: `pjm_load_forecast/data.py`, `pjm_load_forecast/features.py`, `pjm_load_forecast/models.py`, `pjm_load_forecast/evaluation.py`, `pjm_load_forecast/cli.py`.

Each starts as a single line: `"""Module placeholder; implemented in later tasks."""`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml pjm_load_forecast/
git commit -m "chore: scaffold pjm_load_forecast package"
```

---

## Task 2: Sample dataset fixture

**Files:**
- Create: `data/sample.csv`
- Create: `tests/conftest.py`

- [ ] **Step 1: Generate sample CSV**

Write a one-shot script (run, then discard) that emits 720 hours (30 days) of synthetic PJM-shaped load data starting `2018-01-01 00:00:00`. We use synthetic-but-realistic data so tests are deterministic and we don't ship copyrighted real data. Run this in a Python REPL or `python -c`:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
hours = pd.date_range("2018-01-01", periods=720, freq="H")
hour_of_day = hours.hour.to_numpy()
day_of_week = hours.dayofweek.to_numpy()

# Daily peak around 18:00, trough around 04:00
daily = 8000 + 4000 * np.sin((hour_of_day - 10) / 24 * 2 * np.pi)
# Weekend dip
weekly = -1500 * (day_of_week >= 5)
# Slow upward drift
trend = np.linspace(0, 500, 720)
# Noise
noise = rng.normal(0, 300, 720)

mw = 25000 + daily + weekly + trend + noise
df = pd.DataFrame({"Datetime": hours.strftime("%Y-%m-%d %H:%M:%S"), "MW": mw.round(2)})
df.to_csv("data/sample.csv", index=False)
```

- [ ] **Step 2: Verify the CSV opens and has expected shape**

Run:
```bash
python -c "import pandas as pd; df = pd.read_csv('data/sample.csv'); print(df.shape); print(df.head()); print(df.tail())"
```
Expected: shape `(720, 2)`, first row `2018-01-01 00:00:00`, last row `2018-01-30 23:00:00`.

- [ ] **Step 3: Write `tests/conftest.py`**

```python
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
    idx = pd.date_range("2020-01-01", periods=240, freq="H")
    values = 1000 + 100 * np.sin(np.arange(240) / 24 * 2 * np.pi)
    return pd.DataFrame({"MW": values}, index=idx)
```

- [ ] **Step 4: Commit**

```bash
git add data/sample.csv tests/conftest.py
git commit -m "test: add synthetic PJM sample CSV and shared fixtures"
```

---

## Task 3: `data.py` — `load_pjm_csv`

**Files:**
- Modify: `pjm_load_forecast/data.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data.py`:

```python
"""Tests for pjm_load_forecast.data."""
from __future__ import annotations

import pandas as pd
import pytest

from pjm_load_forecast.data import load_pjm_csv, split_by_date


class TestLoadPjmCsv:
    def test_loads_sample_csv(self, sample_csv_path):
        df = load_pjm_csv(sample_csv_path)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["MW"]

    def test_index_is_sorted_datetimeindex(self, sample_csv_path):
        df = load_pjm_csv(sample_csv_path)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.is_monotonic_increasing

    def test_mw_is_float(self, sample_csv_path):
        df = load_pjm_csv(sample_csv_path)
        assert df["MW"].dtype.kind == "f"

    def test_averages_duplicate_timestamps(self, tmp_path):
        csv = tmp_path / "dup.csv"
        csv.write_text(
            "Datetime,MW\n"
            "2020-01-01 00:00:00,100.0\n"
            "2020-01-01 00:00:00,200.0\n"
            "2020-01-01 01:00:00,150.0\n"
        )
        df = load_pjm_csv(csv)
        assert len(df) == 2
        assert df.loc["2020-01-01 00:00:00", "MW"] == 150.0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pjm_csv(tmp_path / "nope.csv")
```

- [ ] **Step 2: Run tests; expect failures**

Run: `pytest tests/test_data.py::TestLoadPjmCsv -v`
Expected: ImportError or AttributeError (`load_pjm_csv` does not exist).

- [ ] **Step 3: Implement `load_pjm_csv`**

Replace `pjm_load_forecast/data.py` with:

```python
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
```

- [ ] **Step 4: Run tests; expect pass**

Run: `pytest tests/test_data.py::TestLoadPjmCsv -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add pjm_load_forecast/data.py tests/test_data.py
git commit -m "feat(data): load_pjm_csv with duplicate-hour averaging"
```

---

## Task 4: `data.py` — `split_by_date`

**Files:**
- Modify: `pjm_load_forecast/data.py`
- Modify: `tests/test_data.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_data.py`:

```python
class TestSplitByDate:
    def test_default_70_15_15_split(self, sample_csv_path):
        df = load_pjm_csv(sample_csv_path)
        train, val, test = split_by_date(df)
        assert len(train) + len(val) + len(test) == len(df)
        # Within 1 row of the requested ratios
        assert abs(len(train) / len(df) - 0.70) < 0.01
        assert abs(len(val) / len(df) - 0.15) < 0.01
        assert abs(len(test) / len(df) - 0.15) < 0.01

    def test_splits_are_chronological(self, sample_csv_path):
        df = load_pjm_csv(sample_csv_path)
        train, val, test = split_by_date(df)
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()

    def test_custom_ratios(self, sample_csv_path):
        df = load_pjm_csv(sample_csv_path)
        train, val, test = split_by_date(df, train_frac=0.5, val_frac=0.25)
        assert abs(len(train) / len(df) - 0.5) < 0.01
        assert abs(len(val) / len(df) - 0.25) < 0.01
        assert abs(len(test) / len(df) - 0.25) < 0.01

    def test_invalid_ratios_raise(self, sample_csv_path):
        df = load_pjm_csv(sample_csv_path)
        with pytest.raises(ValueError):
            split_by_date(df, train_frac=0.7, val_frac=0.4)
```

- [ ] **Step 2: Run; expect failures**

Run: `pytest tests/test_data.py::TestSplitByDate -v`
Expected: ImportError on `split_by_date`.

- [ ] **Step 3: Implement `split_by_date`**

Append to `pjm_load_forecast/data.py`:

```python
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
```

- [ ] **Step 4: Run; expect pass**

Run: `pytest tests/test_data.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add pjm_load_forecast/data.py tests/test_data.py
git commit -m "feat(data): chronological train/val/test splitter"
```

---

## Task 5: `features.py` — calendar features

**Files:**
- Modify: `pjm_load_forecast/features.py`
- Create: `tests/test_features.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_features.py`:

```python
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
        idx = pd.date_range("2020-01-01 00:00:00", periods=3, freq="H")
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
```

- [ ] **Step 2: Run; expect failures**

Run: `pytest tests/test_features.py::TestCalendarFeatures -v`
Expected: ImportError on `add_calendar_features`.

- [ ] **Step 3: Implement `add_calendar_features`**

Replace `pjm_load_forecast/features.py` with:

```python
"""Feature engineering for PJM hourly load forecasting."""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


_HOLIDAY_CACHE: dict[tuple[pd.Timestamp, pd.Timestamp], pd.DatetimeIndex] = {}


def _us_holidays(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    key = (start.normalize(), end.normalize())
    if key not in _HOLIDAY_CACHE:
        cal = USFederalHolidayCalendar()
        _HOLIDAY_CACHE[key] = cal.holidays(start=key[0], end=key[1])
    return _HOLIDAY_CACHE[key]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with calendar features appended.

    Adds: ``hour``, ``dayofweek``, ``month``, ``is_weekend``,
    ``is_holiday`` (US federal holidays).
    """
    out = df.copy()
    idx = out.index
    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["month"] = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    holidays = _us_holidays(idx.min(), idx.max())
    out["is_holiday"] = idx.normalize().isin(holidays).astype(int)
    return out
```

- [ ] **Step 4: Run; expect pass**

Run: `pytest tests/test_features.py::TestCalendarFeatures -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add pjm_load_forecast/features.py tests/test_features.py
git commit -m "feat(features): calendar features (hour/dow/month/weekend/holiday)"
```

---

## Task 6: `features.py` — lag features

**Files:**
- Modify: `pjm_load_forecast/features.py`
- Modify: `tests/test_features.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_features.py`:

```python
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
```

- [ ] **Step 2: Run; expect failures**

Run: `pytest tests/test_features.py::TestLagFeatures -v`
Expected: ImportError on `add_lag_features`.

- [ ] **Step 3: Implement `add_lag_features`**

Append to `pjm_load_forecast/features.py`:

```python
def add_lag_features(
    df: pd.DataFrame,
    lags: Iterable[int],
    column: str = "MW",
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``lag_{k}`` columns for each ``k`` in ``lags``."""
    lags = list(lags)
    if any(l <= 0 for l in lags):
        raise ValueError(f"All lags must be positive; got {lags}")
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[column].shift(lag)
    return out
```

- [ ] **Step 4: Run; expect pass**

Run: `pytest tests/test_features.py::TestLagFeatures -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add pjm_load_forecast/features.py tests/test_features.py
git commit -m "feat(features): configurable lag features"
```

---

## Task 7: `features.py` — rolling features + design matrix

**Files:**
- Modify: `pjm_load_forecast/features.py`
- Modify: `tests/test_features.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_features.py`:

```python
class TestRollingFeatures:
    def test_columns_added(self, synthetic_hourly_series):
        out = add_rolling_features(synthetic_hourly_series, windows=[24, 168])
        assert "rollmean_24" in out.columns
        assert "rollmean_168" in out.columns

    def test_rolling_mean_uses_only_past(self, synthetic_hourly_series):
        out = add_rolling_features(synthetic_hourly_series, windows=[3])
        # rollmean_3 at row i should be mean of rows i-3, i-2, i-1 (past only)
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
        # Largest lag is 24, so we should drop the first 24 rows
        assert len(X) == len(synthetic_hourly_series) - 24

    def test_target_must_exist(self, synthetic_hourly_series):
        with pytest.raises(KeyError):
            build_design_matrix(synthetic_hourly_series, target="nope")
```

- [ ] **Step 2: Run; expect failures**

Run: `pytest tests/test_features.py::TestRollingFeatures tests/test_features.py::TestBuildDesignMatrix -v`
Expected: ImportError on the new functions.

- [ ] **Step 3: Implement both**

Append to `pjm_load_forecast/features.py`:

```python
def add_rolling_features(
    df: pd.DataFrame,
    windows: Iterable[int],
    column: str = "MW",
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``rollmean_{w}`` columns for each window.

    Each rolling mean uses only past values: it is computed on a 1-step
    shifted series, so no future leakage.
    """
    windows = list(windows)
    if any(w <= 0 for w in windows):
        raise ValueError(f"All windows must be positive; got {windows}")
    out = df.copy()
    shifted = out[column].shift(1)
    for w in windows:
        out[f"rollmean_{w}"] = shifted.rolling(window=w, min_periods=w).mean()
    return out


def build_design_matrix(
    df: pd.DataFrame,
    target: str = "MW",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Drop rows with NaNs and return ``(X, y)`` aligned by index.

    The target column is removed from ``X``.
    """
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in DataFrame")
    clean = df.dropna()
    y = clean[target]
    X = clean.drop(columns=[target])
    return X, y
```

- [ ] **Step 4: Run; expect pass**

Run: `pytest tests/test_features.py -v`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add pjm_load_forecast/features.py tests/test_features.py
git commit -m "feat(features): rolling-mean features and design-matrix builder"
```

---

## Task 8: `models.py` — `SeasonalNaive`

**Files:**
- Modify: `pjm_load_forecast/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_models.py`:

```python
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
        # Build a series where lag-168 is meaningful
        idx = pd.date_range("2020-01-01", periods=400, freq="H")
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
```

- [ ] **Step 2: Run; expect failures**

Run: `pytest tests/test_models.py::TestSeasonalNaive -v`
Expected: ImportError on the model classes.

- [ ] **Step 3: Implement `SeasonalNaive` and stub the others**

Replace `pjm_load_forecast/models.py` with:

```python
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
```

(We'll add `LinearModel` and `GradientBoostingModel` in the next tasks; for now keep `__init__.py` happy by adding placeholder classes:)

Append:

```python
class LinearModel:  # implemented in next task
    pass


class GradientBoostingModel:  # implemented in next task
    pass
```

- [ ] **Step 4: Run; expect pass for SeasonalNaive only**

Run: `pytest tests/test_models.py::TestSeasonalNaive -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add pjm_load_forecast/models.py tests/test_models.py
git commit -m "feat(models): SeasonalNaive baseline (lag-168 lookup)"
```

---

## Task 9: `models.py` — `LinearModel`

**Files:**
- Modify: `pjm_load_forecast/models.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_models.py`:

```python
class TestLinearModel:
    def test_recovers_known_slope_on_noiseless_data(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200)})
        y = pd.Series(3.0 * X["a"] - 2.0 * X["b"] + 5.0)
        model = LinearModel(alpha=0.0)
        model.fit(X, y)
        preds = model.predict(X)
        # With alpha=0 on noiseless data, RMSE should be tiny
        rmse = float(np.sqrt(np.mean((preds - y.to_numpy()) ** 2)))
        assert rmse < 1e-6

    def test_predict_shape(self):
        X = pd.DataFrame({"a": np.arange(20.0)})
        y = pd.Series(np.arange(20.0) * 2)
        model = LinearModel()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (20,)

    def test_predict_before_fit_raises(self):
        model = LinearModel()
        with pytest.raises(RuntimeError):
            model.predict(pd.DataFrame({"a": [1.0, 2.0]}))
```

- [ ] **Step 2: Run; expect failures**

Run: `pytest tests/test_models.py::TestLinearModel -v`
Expected: TypeError or AttributeError because `LinearModel` is a placeholder.

- [ ] **Step 3: Implement `LinearModel`**

In `pjm_load_forecast/models.py`, **replace the `LinearModel` placeholder** with:

```python
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
```

- [ ] **Step 4: Run; expect pass**

Run: `pytest tests/test_models.py::TestLinearModel -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add pjm_load_forecast/models.py tests/test_models.py
git commit -m "feat(models): scaled ridge regression"
```

---

## Task 10: `models.py` — `GradientBoostingModel`

**Files:**
- Modify: `pjm_load_forecast/models.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_models.py`:

```python
class TestGradientBoostingModel:
    def test_beats_mean_predictor_on_signal(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"x": rng.uniform(0, 10, 500)})
        y = pd.Series(np.sin(X["x"]) * 5 + rng.normal(0, 0.1, 500))
        model = GradientBoostingModel(max_iter=100, learning_rate=0.05)
        model.fit(X, y)
        preds = model.predict(X)
        gbm_rmse = float(np.sqrt(np.mean((preds - y.to_numpy()) ** 2)))
        mean_rmse = float(np.sqrt(np.mean((y.mean() - y.to_numpy()) ** 2)))
        assert gbm_rmse < 0.5 * mean_rmse

    def test_predict_shape(self):
        X = pd.DataFrame({"a": np.arange(50.0)})
        y = pd.Series(np.arange(50.0) ** 0.5)
        model = GradientBoostingModel(max_iter=20)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (50,)

    def test_predict_before_fit_raises(self):
        model = GradientBoostingModel()
        with pytest.raises(RuntimeError):
            model.predict(pd.DataFrame({"a": [1.0, 2.0]}))
```

- [ ] **Step 2: Run; expect failures**

Run: `pytest tests/test_models.py::TestGradientBoostingModel -v`
Expected: TypeError because of placeholder class.

- [ ] **Step 3: Implement `GradientBoostingModel`**

In `pjm_load_forecast/models.py`, **replace the `GradientBoostingModel` placeholder** with:

```python
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
```

- [ ] **Step 4: Run; expect pass**

Run: `pytest tests/test_models.py -v`
Expected: 9 passed total.

- [ ] **Step 5: Commit**

```bash
git add pjm_load_forecast/models.py tests/test_models.py
git commit -m "feat(models): HistGradientBoosting forecaster"
```

---

## Task 11: `evaluation.py` — metrics

**Files:**
- Modify: `pjm_load_forecast/evaluation.py`
- Create: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_evaluation.py`:

```python
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
        # |0.5| + |0| + |1.0| = 1.5; mean = 0.5
        assert mae(y_true, y_pred) == pytest.approx(0.5)

    def test_rmse(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        assert rmse(y_true, y_pred) == pytest.approx(1.0)

    def test_mape(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        # (10/100 + 20/200) / 2 = (0.1 + 0.1) / 2 = 0.1
        assert mape(y_true, y_pred) == pytest.approx(0.1)

    def test_mape_skips_zero_truth(self):
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([10.0, 110.0])
        # Only the second pair counts: 10/100 = 0.1
        assert mape(y_true, y_pred) == pytest.approx(0.1)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            mae(np.array([1.0, 2.0]), np.array([1.0]))
        with pytest.raises(ValueError):
            rmse(np.array([1.0, 2.0]), np.array([1.0]))
        with pytest.raises(ValueError):
            mape(np.array([1.0, 2.0]), np.array([1.0]))
```

- [ ] **Step 2: Run; expect failures**

Run: `pytest tests/test_evaluation.py::TestMetrics -v`
Expected: ImportError.

- [ ] **Step 3: Implement metrics**

Replace `pjm_load_forecast/evaluation.py` with:

```python
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
```

- [ ] **Step 4: Run; expect pass**

Run: `pytest tests/test_evaluation.py::TestMetrics -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add pjm_load_forecast/evaluation.py tests/test_evaluation.py
git commit -m "feat(evaluation): MAE / RMSE / MAPE metrics"
```

---

## Task 12: `evaluation.py` — `walk_forward_backtest`

**Files:**
- Modify: `pjm_load_forecast/evaluation.py`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_evaluation.py`:

```python
class TestWalkForwardBacktest:
    def _make_df(self):
        # 600 hours, with a 168-hour seasonal pattern + small trend
        idx = pd.date_range("2020-01-01", periods=600, freq="H")
        season = 100 * np.sin(np.arange(600) / 168 * 2 * np.pi)
        trend = np.linspace(0, 5, 600)
        df = pd.DataFrame({"MW": 1000 + season + trend}, index=idx)
        # Add the lag-168 column the SeasonalNaive model needs
        df["lag_168"] = df["MW"].shift(168)
        return df.dropna()

    def test_returns_expected_keys_and_shapes(self):
        df = self._make_df()
        model = SeasonalNaive(lag=168)
        result = walk_forward_backtest(
            model, df, target="MW", train_size=168, horizon=24, step=24
        )
        assert set(result.keys()) >= {"mae", "rmse", "mape", "n_windows", "predictions"}
        assert isinstance(result["predictions"], pd.Series)
        assert result["n_windows"] >= 1

    def test_seasonal_naive_low_error_on_seasonal_data(self):
        df = self._make_df()
        model = SeasonalNaive(lag=168)
        result = walk_forward_backtest(
            model, df, target="MW", train_size=168, horizon=24, step=24
        )
        # With pure 168-period seasonality, lag-168 should be very accurate
        assert result["mape"] < 0.05

    def test_invalid_train_size_raises(self):
        df = self._make_df()
        model = SeasonalNaive(lag=168)
        with pytest.raises(ValueError):
            walk_forward_backtest(model, df, target="MW", train_size=0, horizon=24)
```

- [ ] **Step 2: Run; expect failures**

Run: `pytest tests/test_evaluation.py::TestWalkForwardBacktest -v`
Expected: ImportError on `walk_forward_backtest`.

- [ ] **Step 3: Implement `walk_forward_backtest`**

Append to `pjm_load_forecast/evaluation.py`:

```python
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
```

- [ ] **Step 4: Run; expect pass**

Run: `pytest tests/test_evaluation.py -v`
Expected: 8 passed total.

- [ ] **Step 5: Commit**

```bash
git add pjm_load_forecast/evaluation.py tests/test_evaluation.py
git commit -m "feat(evaluation): expanding-window walk-forward backtest"
```

---

## Task 13: `cli.py`

**Files:**
- Modify: `pjm_load_forecast/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cli.py`:

```python
"""Smoke tests for the CLI."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pjm_load_forecast.cli", *args],
        capture_output=True,
        text=True,
    )


class TestCli:
    def test_evaluate_naive_prints_metrics(self, sample_csv_path):
        proc = _run(
            [
                "evaluate",
                "--data", str(sample_csv_path),
                "--model", "naive",
                "--train-size", "200",
                "--horizon", "24",
                "--step", "48",
            ]
        )
        assert proc.returncode == 0, proc.stderr
        assert "mape" in proc.stdout.lower()

    def test_evaluate_json_output(self, sample_csv_path):
        proc = _run(
            [
                "evaluate",
                "--data", str(sample_csv_path),
                "--model", "naive",
                "--train-size", "200",
                "--horizon", "24",
                "--step", "48",
                "--json",
            ]
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert {"mae", "rmse", "mape", "n_windows"} <= set(payload.keys())

    def test_train_then_predict(self, tmp_path, sample_csv_path):
        model_path = tmp_path / "model.pkl"
        pred_path = tmp_path / "pred.csv"
        train_proc = _run(
            [
                "train",
                "--data", str(sample_csv_path),
                "--model", "linear",
                "--out", str(model_path),
            ]
        )
        assert train_proc.returncode == 0, train_proc.stderr
        assert model_path.exists()

        predict_proc = _run(
            [
                "predict",
                "--model-path", str(model_path),
                "--data", str(sample_csv_path),
                "--out", str(pred_path),
            ]
        )
        assert predict_proc.returncode == 0, predict_proc.stderr
        assert pred_path.exists()
        # Should have header and at least one prediction row
        lines = pred_path.read_text().splitlines()
        assert lines[0].lower().startswith("datetime,prediction") or "prediction" in lines[0].lower()
        assert len(lines) > 1

    def test_unknown_subcommand_returns_nonzero(self):
        proc = _run(["wat"])
        assert proc.returncode != 0
```

- [ ] **Step 2: Run; expect failures**

Run: `pytest tests/test_cli.py -v`
Expected: failures because `cli.py` is still a placeholder.

- [ ] **Step 3: Implement `cli.py`**

Replace `pjm_load_forecast/cli.py` with:

```python
"""Command-line interface for pjm_load_forecast."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import joblib
import pandas as pd

from pjm_load_forecast.data import load_pjm_csv
from pjm_load_forecast.evaluation import walk_forward_backtest
from pjm_load_forecast.features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    build_design_matrix,
)
from pjm_load_forecast.models import (
    GradientBoostingModel,
    LinearModel,
    SeasonalNaive,
)


_MODELS = {
    "naive": lambda: SeasonalNaive(lag=168),
    "linear": lambda: LinearModel(alpha=1.0),
    "gbm": lambda: GradientBoostingModel(),
}


def _featurize(df: pd.DataFrame) -> pd.DataFrame:
    df = add_calendar_features(df)
    df = add_lag_features(df, lags=[1, 24, 168])
    df = add_rolling_features(df, windows=[24, 168])
    return df


def _cmd_evaluate(args: argparse.Namespace) -> int:
    df = _featurize(load_pjm_csv(args.data)).dropna()
    model = _MODELS[args.model]()
    result = walk_forward_backtest(
        model,
        df,
        target="MW",
        train_size=args.train_size,
        horizon=args.horizon,
        step=args.step,
    )
    payload = {
        "model": args.model,
        "mae": round(result["mae"], 4),
        "rmse": round(result["rmse"], 4),
        "mape": round(result["mape"], 6),
        "n_windows": result["n_windows"],
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print(f"model     : {payload['model']}")
        print(f"n_windows : {payload['n_windows']}")
        print(f"mae       : {payload['mae']}")
        print(f"rmse      : {payload['rmse']}")
        print(f"mape      : {payload['mape']:.4%}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    df = _featurize(load_pjm_csv(args.data))
    X, y = build_design_matrix(df, target="MW")
    model = _MODELS[args.model]()
    model.fit(X, y)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": list(X.columns)}, out)
    print(f"saved model to {out}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    bundle = joblib.load(args.model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df = _featurize(load_pjm_csv(args.data))
    X, _ = build_design_matrix(df, target="MW")
    # Re-order columns to match training
    X = X[feature_cols]
    preds = model.predict(X)
    out_df = pd.DataFrame({"Datetime": X.index, "prediction": preds})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"wrote {len(out_df)} predictions to {out_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pjm-forecast", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("evaluate", help="walk-forward backtest")
    p_eval.add_argument("--data", required=True)
    p_eval.add_argument("--model", choices=_MODELS.keys(), required=True)
    p_eval.add_argument("--train-size", type=int, default=24 * 30 * 6)
    p_eval.add_argument("--horizon", type=int, default=24)
    p_eval.add_argument("--step", type=int, default=168)
    p_eval.add_argument("--json", action="store_true")
    p_eval.set_defaults(func=_cmd_evaluate)

    p_train = sub.add_parser("train", help="fit a model on the full series")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--model", choices=_MODELS.keys(), required=True)
    p_train.add_argument("--out", required=True)
    p_train.set_defaults(func=_cmd_train)

    p_pred = sub.add_parser("predict", help="predict from a saved model")
    p_pred.add_argument("--model-path", required=True)
    p_pred.add_argument("--data", required=True)
    p_pred.add_argument("--out", required=True)
    p_pred.set_defaults(func=_cmd_predict)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run; expect pass**

Run: `pytest tests/test_cli.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add pjm_load_forecast/cli.py tests/test_cli.py
git commit -m "feat(cli): pjm-forecast {evaluate,train,predict} subcommands"
```

---

## Task 14: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README**

Create `README.md`:

````markdown
# pjm_load_forecast

Hourly electricity load forecasting for the PJM grid — final project for
**ORIE 5270 (Cornell, Spring 2026)**.

## Purpose

Forecast next-24-hour electricity load (MW) for a PJM zone from a single
historical hourly load CSV. The package ships three models of increasing
sophistication and a walk-forward backtest harness so they can be compared
on a held-out tail of the series.

## Dataset

[PJM Interconnection](https://dataminer2.pjm.com/list) publishes hourly
load per zone. The same series is mirrored as the *PJM Hourly Energy
Consumption* dataset (CSV per zone, columns: ``Datetime``, ``MW``).

A ~720-row synthetic-but-PJM-shaped sample is bundled at
``data/sample.csv`` so the tests and the demo notebook run without
network access. To work with a real zone, point ``--data`` at any PJM
hourly load CSV with the same two-column schema.

## Install

```bash
git clone <this repo>
cd orie5270-final-project
pip install -e ".[dev]"
```

## Quick start

```bash
# Backtest the gradient-boosting model on the bundled sample
pjm-forecast evaluate --data data/sample.csv --model gbm \
  --train-size 200 --horizon 24 --step 48

# Train + persist a model, then predict
pjm-forecast train   --data data/sample.csv --model linear --out model.pkl
pjm-forecast predict --model-path model.pkl --data data/sample.csv --out preds.csv
```

## Library usage

```python
from pjm_load_forecast import (
    load_pjm_csv, add_calendar_features, add_lag_features,
    add_rolling_features, build_design_matrix,
    GradientBoostingModel, walk_forward_backtest,
)

df = load_pjm_csv("data/sample.csv")
df = add_calendar_features(df)
df = add_lag_features(df, lags=[1, 24, 168])
df = add_rolling_features(df, windows=[24, 168])

result = walk_forward_backtest(
    GradientBoostingModel(), df.dropna(),
    target="MW", train_size=200, horizon=24, step=48,
)
print(result["mape"], result["rmse"])
```

## Project structure

```
pjm_load_forecast/
├── pjm_load_forecast/
│   ├── data.py          # load + split CSV
│   ├── features.py      # calendar + lag + rolling features
│   ├── models.py        # SeasonalNaive, LinearModel, GradientBoostingModel
│   ├── evaluation.py    # MAPE/RMSE/MAE + walk-forward backtest
│   └── cli.py           # pjm-forecast {evaluate,train,predict}
├── tests/               # pytest suite (>80% coverage)
├── data/sample.csv
├── notebooks/demo.ipynb
└── pyproject.toml
```

## Testing

```bash
make test           # pytest with --cov-fail-under=80
make coverage       # HTML coverage report at htmlcov/index.html
```

## Methodology

- **Features.** Calendar (hour, day-of-week, month, weekend, US federal
  holiday) plus past-value lags (1h, 24h, 168h) and trailing rolling
  means (24h, 168h). All rolling features are computed on a 1-step
  shifted series, so there's no future leakage.
- **Models.** ``SeasonalNaive`` predicts the value 168 hours ago — a
  strong, hard-to-beat baseline for hourly load. ``LinearModel`` is
  scaled ridge regression. ``GradientBoostingModel`` is scikit-learn's
  ``HistGradientBoostingRegressor``.
- **Evaluation.** Expanding-window walk-forward backtest: refit the
  model every ``step`` hours (default 168), forecast the next
  ``horizon`` hours (default 24), and aggregate MAE/RMSE/MAPE over all
  out-of-sample predictions.

## License

MIT.
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with install / usage / methodology"
```

---

## Task 15: Makefile + scripts/download_sample.py

**Files:**
- Create: `Makefile`
- Create: `scripts/download_sample.py`

- [ ] **Step 1: Write the Makefile**

Create `Makefile`:

```makefile
.PHONY: install test coverage lint clean

install:
	pip install -e ".[dev]"

test:
	pytest --cov=pjm_load_forecast --cov-report=term-missing --cov-fail-under=80

coverage:
	pytest --cov=pjm_load_forecast --cov-report=html
	@echo "Open htmlcov/index.html in a browser"

lint:
	ruff check pjm_load_forecast tests

clean:
	rm -rf .pytest_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
```

- [ ] **Step 2: Write `scripts/download_sample.py`**

Create `scripts/download_sample.py`:

```python
"""Download a PJM hourly load CSV for one zone.

This helper is for users who want the full multi-year series. The bundled
``data/sample.csv`` is sufficient to run the tests and the demo notebook.

Usage:
    python scripts/download_sample.py --zone PJME --out data/PJME_hourly.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

# Public mirrors of the PJM hourly series. PJM Data Miner 2 itself requires
# an API key; for reproducibility the README points users at the long-mirrored
# Kaggle dataset, which they can download manually if they prefer.
_MIRROR_TEMPLATE = (
    "https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/{zone}_hourly.csv"
)
_KNOWN_ZONES = {"PJME", "PJMW", "AEP", "COMED", "DAYTON", "DEOK", "DOM", "DUQ", "EKPC", "FE", "NI"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zone", required=True, help=f"one of {sorted(_KNOWN_ZONES)}")
    parser.add_argument("--out", required=True, help="output CSV path")
    args = parser.parse_args(argv)

    if args.zone not in _KNOWN_ZONES:
        print(f"unknown zone {args.zone!r}; known: {sorted(_KNOWN_ZONES)}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    url = _MIRROR_TEMPLATE.format(zone=args.zone)
    print(f"downloading {url} -> {out}")
    urlretrieve(url, out)  # noqa: S310 - user-provided zone, well-known mirror
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Verify the Makefile runs**

Run: `make test`
Expected: full pytest run with coverage report; ≥80% coverage; exit 0.

- [ ] **Step 4: Commit**

```bash
git add Makefile scripts/download_sample.py
git commit -m "chore: add Makefile and PJM CSV downloader script"
```

---

## Task 16: Demo notebook

**Files:**
- Create: `notebooks/demo.ipynb`

- [ ] **Step 1: Write the notebook**

Create `notebooks/demo.ipynb` as a JSON file. Use this exact content:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# PJM Hourly Load Forecasting — Demo\n",
    "\n",
    "End-to-end walkthrough using the bundled sample CSV: load the data, build features, fit each of the three models, and compare them via walk-forward backtesting."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "from pjm_load_forecast import (\n",
    "    load_pjm_csv, add_calendar_features, add_lag_features,\n",
    "    add_rolling_features, build_design_matrix,\n",
    "    SeasonalNaive, LinearModel, GradientBoostingModel,\n",
    "    walk_forward_backtest,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = load_pjm_csv('../data/sample.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "feats = add_calendar_features(df)\n",
    "feats = add_lag_features(feats, lags=[1, 24, 168])\n",
    "feats = add_rolling_features(feats, windows=[24, 168])\n",
    "feats = feats.dropna()\n",
    "feats.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Backtest each model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "models = {\n",
    "    'naive':  SeasonalNaive(lag=168),\n",
    "    'linear': LinearModel(alpha=1.0),\n",
    "    'gbm':    GradientBoostingModel(max_iter=200, learning_rate=0.05),\n",
    "}\n",
    "rows = []\n",
    "for name, model in models.items():\n",
    "    r = walk_forward_backtest(model, feats, target='MW', train_size=200, horizon=24, step=48)\n",
    "    rows.append({'model': name, 'mae': r['mae'], 'rmse': r['rmse'], 'mape': r['mape'], 'n_windows': r['n_windows']})\n",
    "pd.DataFrame(rows)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Verify the notebook is valid JSON**

Run: `python -c "import json; json.load(open('notebooks/demo.ipynb'))"`
Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add notebooks/demo.ipynb
git commit -m "docs: add end-to-end demo notebook"
```

---

## Task 17: Coverage gate + final cleanup

**Files:**
- (no new files; verify only)

- [ ] **Step 1: Run the full test suite with coverage**

Run: `pytest --cov=pjm_load_forecast --cov-report=term-missing --cov-fail-under=80`
Expected: all tests pass; coverage ≥ 80%.

If coverage is < 80%, identify the uncovered lines (printed by
``--cov-report=term-missing``), add a targeted test for each significant
gap, and re-run. Do not silence with ``# pragma: no cover`` unless the
line truly cannot be exercised (e.g., the ``if __name__ == "__main__":``
guard, which is already excluded in ``pyproject.toml``).

- [ ] **Step 2: Verify the CLI installs as a console script**

Run: `which pjm-forecast && pjm-forecast evaluate --data data/sample.csv --model naive --train-size 200 --horizon 24 --step 48`
Expected: prints metrics with non-zero ``n_windows`` and finite ``mape``.

- [ ] **Step 3: Verify the package re-exports work**

Run: `python -c "import pjm_load_forecast as p; print(p.__version__); print(p.__all__)"`
Expected: prints ``0.1.0`` and the list of public names.

- [ ] **Step 4: Final commit if anything changed**

If steps 1–3 required follow-up changes, commit them:

```bash
git add -A
git commit -m "test: bring coverage above 80% threshold"
```

If nothing changed, skip this step.

---

## Self-Review

**Spec coverage:**
- §1 Purpose → covered by entire plan.
- §2 Dataset → Task 2 (sample CSV), Task 14 (README), Task 15 (download script).
- §3 Forecasting task (univariate, 24h horizon) → Task 12 (default `horizon=24`), Task 13 CLI (`--horizon 24` default).
- §4 Architecture (5 modules + tests + data + notebook + scripts + pyproject + README + Makefile) → Tasks 1, 3–13, 14, 15, 16.
- §5 Data pipeline (load, dedup, split, features, design matrix) → Tasks 3, 4, 5, 6, 7.
- §6 Models (SeasonalNaive, LinearModel, GradientBoostingModel) → Tasks 8, 9, 10.
- §7 Evaluation (MAPE/RMSE/MAE + walk-forward) → Tasks 11, 12.
- §8 CLI (train/predict/evaluate, JSON output) → Task 13.
- §9 Testing (≥85% target, hand-checked tests) → every implementation task is preceded by failing tests with hand-computed expectations; Task 17 gates on ≥80%.
- §10 Dependencies (numpy, pandas, sklearn, joblib + dev) → Task 1.
- §11 Out of scope → enforced by absence; no weather/Optuna/quantile work appears.
- §12 Risks → schema-pinned via fixture (Task 2); honest reporting in README (Task 14); hand-computed test values throughout.

**Placeholder scan:** Searched for "TBD", "TODO", "implement later", "appropriate error handling", "similar to Task". None present.

**Type/name consistency:** `SeasonalNaive(lag=168)` with `lag_{lag}` column lookup is consistent across Tasks 8, 12, 13. `walk_forward_backtest` signature matches across Tasks 12 and 13 (kwargs: `target`, `train_size`, `horizon`, `step`). `build_design_matrix(df, target=...)` consistent in Tasks 7 and 13. `_MODELS` dict keys (`naive`/`linear`/`gbm`) are referenced consistently in Task 13 CLI and Task 16 notebook. `feature_cols` is captured during training (Task 13 `_cmd_train`) and re-applied during prediction (`_cmd_predict`), preventing column-order mismatches.

No issues found.
