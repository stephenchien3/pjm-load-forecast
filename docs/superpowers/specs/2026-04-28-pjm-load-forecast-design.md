# PJM Hourly Load Forecasting — Design

**Status:** Approved
**Date:** 2026-04-28
**Course:** ORIE 5270 — Final Project

## 1. Purpose

Build a small, well-tested Python package that forecasts hourly electricity load
on the PJM grid. The package ships three forecasters of increasing
sophistication and an evaluation harness that compares them via walk-forward
backtesting.

The deliverable targets the ORIE 5270 final project rubric:

- Real dataset (PJM hourly load).
- Installable Python package with a CLI.
- README explaining purpose, dataset, install, and usage.
- ≥ 80% unit-test coverage.
- Clean module boundaries and detailed docstrings.

## 2. Dataset

PJM Interconnection publishes hourly load (megawatt-hours) per zone via
[Data Miner 2](https://dataminer2.pjm.com/list). The same series is mirrored on
Kaggle as the *PJM Hourly Energy Consumption* dataset, covering roughly
2002–2018 with one CSV per zone (PJME, PJMW, AEP, COMED, DAYTON, DEOK, DOM,
DUQ, EKPC, FE, NI). Each row has two columns:

```
Datetime,MW
2002-12-31 01:00:00,26498.0
```

The package will:

- Accept any of these CSVs (path passed in, no network required at runtime).
- Bundle a small (~500 row) sample CSV under `data/sample.csv` for tests and
  the README demo.
- Provide a `scripts/download_sample.py` helper that fetches one zone's CSV
  from a public mirror for users who want the full series.

## 3. Forecasting Task

Given hourly load $y_1, \ldots, y_t$, predict the next $h$ hours
$\hat{y}_{t+1}, \ldots, \hat{y}_{t+h}$. Default horizon is **24 hours**.
Inputs are univariate (one zone at a time); the design intentionally avoids
exogenous weather data to keep scope tight.

## 4. Architecture

```
pjm_load_forecast/
├── pjm_load_forecast/
│   ├── __init__.py        # public re-exports
│   ├── data.py            # load CSV, parse timestamps, train/val/test split
│   ├── features.py        # calendar + lag + rolling feature builders
│   ├── models.py          # SeasonalNaive, LinearModel, GradientBoostingModel
│   ├── evaluation.py      # metrics + walk-forward backtest
│   └── cli.py             # `pjm-forecast {train,predict,evaluate}`
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_evaluation.py
│   └── test_cli.py
├── data/sample.csv        # ~500-row fixture (also used in README demo)
├── notebooks/demo.ipynb   # end-to-end walkthrough
├── scripts/download_sample.py
├── pyproject.toml
├── README.md
├── Makefile               # install / lint / test / coverage
└── .gitignore
```

### Module responsibilities

| Module | Public API | Depends on |
|---|---|---|
| `data` | `load_pjm_csv(path) -> pd.DataFrame`, `split_by_date(df, ...) -> (train, val, test)` | pandas |
| `features` | `add_calendar_features(df)`, `add_lag_features(df, lags)`, `add_rolling_features(df, windows)`, `build_design_matrix(df) -> (X, y)` | pandas, numpy |
| `models` | `SeasonalNaive`, `LinearModel`, `GradientBoostingModel` (all with `.fit(X, y)`, `.predict(X) -> np.ndarray`) | scikit-learn, numpy |
| `evaluation` | `mape`, `rmse`, `mae`, `walk_forward_backtest(model, df, ...) -> dict` | numpy, pandas |
| `cli` | argparse entry point — wires the above together | the modules above |

Each module has one clear job, depends only on lower layers, and is testable
in isolation.

## 5. Data Pipeline

1. **Load** — read CSV, parse `Datetime` column, sort by time, set as index,
   coerce duplicates (PJM has occasional DST repeat hours) by averaging.
2. **Resample** — assert hourly frequency; forward-fill at most 1 missing hour
   and drop longer gaps.
3. **Split** — chronological 70/15/15 train/val/test by date (no shuffling).
4. **Feature build** — calendar features (hour, day-of-week, month, is-weekend,
   is-holiday via `pandas.tseries.holiday.USFederalHolidayCalendar`), lag
   features (1, 24, 168 hours), rolling features (24h mean, 168h mean).
5. **Design matrix** — drop rows where any lag/rolling value is NaN; return
   `(X, y)` aligned by index.

## 6. Models

All models implement `fit(X, y)` and `predict(X) -> np.ndarray`. They are
deliberately simple so behavior can be unit-tested.

1. **`SeasonalNaive`** — predicts $\hat{y}_t = y_{t-168}$ (same hour, one
   week ago). Implementation just looks up the lag-168 column. No training
   needed but `fit` is provided for a uniform interface.

2. **`LinearModel`** — `sklearn.linear_model.Ridge` on the full design matrix.
   Standardizes features inside a `Pipeline`. Alpha is configurable; default
   1.0.

3. **`GradientBoostingModel`** — `sklearn.ensemble.HistGradientBoostingRegressor`
   on the full design matrix. Default 200 trees, learning rate 0.05, max depth
   8. No extra dependency beyond scikit-learn.

## 7. Evaluation

- **Metrics:** MAPE, RMSE, MAE — implemented from scratch with hand-checked
  unit tests. Inputs are 1-D arrays of equal length; mismatched shapes raise
  `ValueError`.
- **Walk-forward backtest:** expanding window. Starting from the end of the
  train set, refit the model every `step` hours (default 168 = weekly) and
  predict the next `horizon` hours. Concatenate predictions, compute metrics
  on the held-out region. Returns a dict like
  `{"mape": ..., "rmse": ..., "mae": ..., "n_windows": ..., "predictions": pd.Series}`.

The README will report the headline numbers from running this on the bundled
PJME zone, with the explicit goal of showing GBM beats SeasonalNaive by a
meaningful margin (target: ≥ 30% MAPE reduction).

## 8. CLI

```
pjm-forecast train    --data DATA.csv --model {naive,linear,gbm} --out MODEL.pkl
pjm-forecast predict  --model MODEL.pkl --data DATA.csv --horizon 24 --out PRED.csv
pjm-forecast evaluate --data DATA.csv --model {naive,linear,gbm} [--step 168] [--horizon 24]
```

Argparse-based. Models persist via `joblib.dump` / `joblib.load`. `evaluate`
prints metrics as a table and writes them to stdout as JSON when
`--json` is passed.

## 9. Testing Strategy

Target ≥ 85% line coverage (rubric requires ≥ 80%).

- **`test_data.py`** — loads fixture CSV, asserts dtypes, index, sort order;
  tests duplicate-hour averaging; tests split sizes and date ordering.
- **`test_features.py`** — calendar features have correct values for known
  dates; lag features equal hand-computed shifts; rolling features match
  hand-computed means; design matrix has no NaNs.
- **`test_models.py`** — each model fits a tiny synthetic dataset and produces
  predictions of correct shape; SeasonalNaive predicts the lag-168 column
  exactly; Ridge with `alpha=0` on noiseless linear data recovers known slope
  within tolerance; GBM beats mean prediction on the same noiseless data.
- **`test_evaluation.py`** — metrics computed on hand-picked inputs equal
  hand-computed expected values; mismatched shapes raise `ValueError`;
  backtest returns expected number of windows on a synthetic series.
- **`test_cli.py`** — invokes each subcommand via `subprocess.run` against the
  bundled sample CSV and asserts exit code 0 + expected output files.

`pytest --cov=pjm_load_forecast --cov-report=term-missing --cov-fail-under=80`
runs in CI / `make test`.

## 10. Dependencies

Runtime: `numpy`, `pandas`, `scikit-learn`, `joblib`.
Dev: `pytest`, `pytest-cov`, `ruff` (lint).

No deep-learning dependencies. No XGBoost / LightGBM. Keeps install fast,
tests deterministic, and grading reproducible.

## 11. Out of Scope

- Multivariate models with weather / temperature features.
- Real-time streaming inference (the lecture motivates streams, but for this
  project we forecast on bounded historical CSVs).
- Hyperparameter tuning. Defaults are chosen and documented; Optuna / GridSearch
  are explicitly not included.
- Probabilistic forecasts / quantile regression.

## 12. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| PJM CSV format changes upstream | Pin to the bundled sample for tests; document expected schema in README. |
| GBM doesn't beat SeasonalNaive | Lag-168 is genuinely strong for load; if GBM ties, that's still a valid finding to report. README will state honest numbers either way. |
| Test coverage misses edge cases | Hand-compute expected values rather than golden-file comparisons; assert on shapes and values, not just non-error. |
| Notebook drifts from package API | Notebook only uses public API; CLI smoke test catches breaks at the boundary. |
