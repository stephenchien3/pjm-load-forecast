# pjm_load_forecast

Hourly electricity load forecasting on the PJM grid. Final project for
**ORIE 5270 (Cornell, Spring 2026)**.

## Purpose

Forecast next-hour electricity load (megawatts) for a PJM zone from a
historical hourly load CSV, and compare a strong seasonal baseline
against a gradient-boosting model via walk-forward backtesting.

## Dataset

[PJM Interconnection Data Miner 2](https://dataminer2.pjm.com/list)
publishes hourly load per zone. The same series is mirrored as the
*PJM Hourly Energy Consumption* dataset (one CSV per zone with columns
`Datetime`, `<ZONE>_MW`).

This repo ships ~16 years (Dec 2002 – Aug 2018, 145 366 hourly rows) of
the **PJME** zone at `data/PJME_hourly.csv`.

## Install

```bash
git clone <this repo>
cd orie5270-final-project
pip install -e ".[dev]"
```

## Usage

```python
from pjm_load_forecast import (
    load_pjm_csv, build_features,
    SeasonalNaive, GradientBoostingModel,
    walk_forward_backtest,
)

df = build_features(load_pjm_csv("data/PJME_hourly.csv"))

for name, model in [
    ("naive", SeasonalNaive(lag=168)),
    ("gbm",   GradientBoostingModel()),
]:
    r = walk_forward_backtest(
        model, df, train_size=8760, horizon=24, step=720,
    )
    print(f"{name:6s}  MAPE={r['mape']:.2%}  RMSE={r['rmse']:.0f} MW  "
          f"({r['n_windows']} windows)")
```

## Results

On 16 years of PJME data with a 1-year initial train window, 24-hour
forecast horizon, and monthly refits (190 backtest windows):

| Model | MAPE | RMSE (MW) |
|---|---|---|
| `SeasonalNaive` (lag-168) | 9.43% | 4 427 |
| `GradientBoostingModel`   | **1.03%** | **480** |

GBM beats the seasonal-naive baseline by **~9× on MAPE**. The naive
baseline is the same hour from one week ago — already a strong signal
for hourly load — and gradient boosting captures the residual structure
left over (calendar effects, recent trend, intraday autocorrelation).

## Methodology

- **Features**: hour of day, day of week, month, weekend flag, plus
  past-value lags at 1 h, 24 h, and 168 h.
- **Models**: `SeasonalNaive(lag=168)` (predicts the value one week ago)
  and `HistGradientBoostingRegressor` (200 trees, lr 0.05, depth 8).
- **Evaluation**: expanding-window walk-forward backtest. Refit every
  `step` hours, forecast the next `horizon` hours, aggregate
  MAE/RMSE/MAPE over all out-of-sample predictions.

## Tests

```bash
pytest
```

Runs the full suite (~20 tests) with coverage; the package currently
sits at 100 % line coverage and the suite gates on ≥ 80 %.

## Project layout

```
pjm_load_forecast/
├── pjm_load_forecast/__init__.py   # all the code
├── tests/test_forecast.py
├── data/PJME_hourly.csv
├── pyproject.toml
└── README.md
```

## License

MIT.
