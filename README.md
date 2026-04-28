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
Consumption* dataset (CSV per zone, columns: `Datetime`, `MW`).

A 720-row synthetic-but-PJM-shaped sample is bundled at
`data/sample.csv` so the tests and the demo notebook run without
network access. To work with a real zone, point `--data` at any PJM
hourly load CSV with the same two-column schema, or run
`python scripts/download_sample.py --zone PJME --out data/PJME_hourly.csv`.

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
  shifted series, so there is no future leakage.
- **Models.** `SeasonalNaive` predicts the value 168 hours ago — a
  strong, hard-to-beat baseline for hourly load. `LinearModel` is
  scaled ridge regression. `GradientBoostingModel` is scikit-learn's
  `HistGradientBoostingRegressor`.
- **Evaluation.** Expanding-window walk-forward backtest: refit the
  model every `step` hours (default 168), forecast the next
  `horizon` hours (default 24), and aggregate MAE/RMSE/MAPE over all
  out-of-sample predictions.

## License

MIT.
