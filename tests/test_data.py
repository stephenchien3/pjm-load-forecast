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
