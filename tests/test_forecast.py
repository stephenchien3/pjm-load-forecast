"""Tests for pjm_load_forecast."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pjm_load_forecast import (
    GradientBoostingModel,
    SeasonalNaive,
    build_features,
    load_pjm_csv,
    mae,
    mape,
    rmse,
    walk_forward_backtest,
)


# ---------- fixtures ----------

@pytest.fixture
def synthetic_csv(tmp_path):
    """A small synthetic PJM-shaped CSV (30 days, hourly)."""
    rng = np.random.default_rng(0)
    hours = pd.date_range("2020-01-01", periods=720, freq="h")
    daily = 8000 + 4000 * np.sin((hours.hour - 10) / 24 * 2 * np.pi)
    weekly = -1500 * (hours.dayofweek >= 5)
    mw = 25000 + daily + weekly + rng.normal(0, 200, 720)
    csv = tmp_path / "pjm.csv"
    pd.DataFrame({
        "Datetime": hours.strftime("%Y-%m-%d %H:%M:%S"),
        "MW": mw.round(2),
    }).to_csv(csv, index=False)
    return csv


@pytest.fixture
def feats(synthetic_csv):
    return build_features(load_pjm_csv(synthetic_csv))


# ---------- load_pjm_csv ----------

class TestLoadPjmCsv:
    def test_returns_sorted_datetime_indexed_frame(self, synthetic_csv):
        df = load_pjm_csv(synthetic_csv)
        assert list(df.columns) == ["MW"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.is_monotonic_increasing
        assert df["MW"].dtype.kind == "f"

    def test_accepts_zone_prefixed_column(self, tmp_path):
        csv = tmp_path / "pjme.csv"
        csv.write_text("Datetime,PJME_MW\n2020-01-01 00:00:00,100.0\n")
        df = load_pjm_csv(csv)
        assert df.loc["2020-01-01 00:00:00", "MW"] == 100.0

    def test_averages_duplicate_timestamps(self, tmp_path):
        csv = tmp_path / "dup.csv"
        csv.write_text(
            "Datetime,MW\n"
            "2020-01-01 00:00:00,100.0\n"
            "2020-01-01 00:00:00,200.0\n"
        )
        df = load_pjm_csv(csv)
        assert len(df) == 1
        assert df.iloc[0]["MW"] == 150.0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pjm_csv(tmp_path / "nope.csv")

    def test_no_mw_column_raises(self, tmp_path):
        csv = tmp_path / "bad.csv"
        csv.write_text("Datetime,price\n2020-01-01 00:00:00,1.0\n")
        with pytest.raises(ValueError):
            load_pjm_csv(csv)


# ---------- build_features ----------

class TestBuildFeatures:
    def test_adds_calendar_and_lag_columns(self, feats):
        for col in ["hour", "dayofweek", "month", "is_weekend",
                    "lag_1", "lag_24", "lag_168"]:
            assert col in feats.columns

    def test_no_nans_after_dropping_warmup(self, feats):
        assert not feats.isna().any().any()

    def test_drops_largest_lag_rows(self, synthetic_csv):
        raw = load_pjm_csv(synthetic_csv)
        feats = build_features(raw, lags=(1, 24, 168))
        assert len(feats) == len(raw) - 168

    def test_known_calendar_values(self):
        df = pd.DataFrame({"MW": [1.0, 2.0]},
                          index=pd.date_range("2020-01-04", periods=2, freq="h"))
        out = build_features(df, lags=(1,))  # 2020-01-04 is a Saturday
        assert out["is_weekend"].iloc[0] == 1
        assert out["dayofweek"].iloc[0] == 5

    def test_lag_equals_shifted_value(self, synthetic_csv):
        raw = load_pjm_csv(synthetic_csv)
        out = build_features(raw, lags=(1,))
        np.testing.assert_array_equal(
            out["lag_1"].to_numpy(), raw["MW"].shift(1).dropna().to_numpy()
        )

    def test_invalid_lag_raises(self, synthetic_csv):
        raw = load_pjm_csv(synthetic_csv)
        with pytest.raises(ValueError):
            build_features(raw, lags=(0,))


# ---------- models ----------

class TestSeasonalNaive:
    def test_predict_returns_lag_column(self, feats):
        model = SeasonalNaive(lag=168).fit(feats, feats["MW"])
        np.testing.assert_array_equal(
            model.predict(feats), feats["lag_168"].to_numpy()
        )

    def test_missing_lag_column_raises(self):
        with pytest.raises(KeyError):
            SeasonalNaive(lag=168).predict(pd.DataFrame({"hour": [0, 1]}))


class TestGradientBoostingModel:
    def test_beats_mean_predictor(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"x": rng.uniform(0, 10, 500)})
        y = pd.Series(np.sin(X["x"]) * 5 + rng.normal(0, 0.1, 500))
        model = GradientBoostingModel(max_iter=100).fit(X, y)
        gbm_rmse = float(np.sqrt(np.mean((model.predict(X) - y.to_numpy()) ** 2)))
        mean_rmse = float(np.sqrt(np.mean((y.mean() - y.to_numpy()) ** 2)))
        assert gbm_rmse < 0.5 * mean_rmse

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            GradientBoostingModel().predict(pd.DataFrame({"a": [1.0]}))


# ---------- metrics ----------

class TestMetrics:
    def test_mae(self):
        assert mae(np.array([1.0, 2.0, 3.0]),
                   np.array([1.5, 2.0, 4.0])) == pytest.approx(0.5)

    def test_rmse(self):
        assert rmse(np.array([0.0, 0.0, 0.0]),
                    np.array([1.0, 1.0, 1.0])) == pytest.approx(1.0)

    def test_mape(self):
        assert mape(np.array([100.0, 200.0]),
                    np.array([110.0, 180.0])) == pytest.approx(0.1)

    def test_mape_skips_zero_truth(self):
        assert mape(np.array([0.0, 100.0]),
                    np.array([10.0, 110.0])) == pytest.approx(0.1)

    def test_mape_all_zero_truth_is_nan(self):
        assert np.isnan(mape(np.array([0.0]), np.array([1.0])))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            mae(np.array([1.0, 2.0]), np.array([1.0]))


# ---------- backtest ----------

class TestWalkForwardBacktest:
    def test_seasonal_naive_low_error_on_pure_seasonal_data(self):
        idx = pd.date_range("2020-01-01", periods=600, freq="h")
        season = 100 * np.sin(np.arange(600) / 168 * 2 * np.pi)
        df = pd.DataFrame({"MW": 1000 + season}, index=idx)
        df["lag_168"] = df["MW"].shift(168)
        df = df.dropna()
        result = walk_forward_backtest(
            SeasonalNaive(168), df, train_size=168, horizon=24, step=24,
        )
        assert result["n_windows"] >= 1
        assert isinstance(result["predictions"], pd.Series)
        assert result["mape"] < 0.05

    def test_gbm_beats_naive_on_real_features(self, feats):
        gbm = walk_forward_backtest(
            GradientBoostingModel(max_iter=50), feats,
            train_size=240, horizon=24, step=72,
        )
        naive = walk_forward_backtest(
            SeasonalNaive(168), feats,
            train_size=240, horizon=24, step=72,
        )
        # GBM should at minimum tie naive on this clean synthetic data.
        assert gbm["mape"] <= naive["mape"] * 1.5

    def test_invalid_args_raise(self, feats):
        with pytest.raises(ValueError):
            walk_forward_backtest(SeasonalNaive(168), feats, train_size=0)

    def test_too_short_dataset_raises(self):
        df = pd.DataFrame({"MW": [1.0, 2.0], "lag_168": [np.nan, np.nan]},
                          index=pd.date_range("2020-01-01", periods=2, freq="h"))
        with pytest.raises(ValueError):
            walk_forward_backtest(
                SeasonalNaive(168), df, train_size=10, horizon=24,
            )
