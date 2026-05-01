"""Generate docs/data.json: last 30 days of actual + both models' predictions."""
from __future__ import annotations

import json
from pathlib import Path

from pjm_load_forecast import (
    GradientBoostingModel,
    SeasonalNaive,
    build_features,
    load_pjm_csv,
    mae,
    mape,
    rmse,
)

ROOT = Path(__file__).resolve().parent.parent
df = build_features(load_pjm_csv(ROOT / "data" / "PJME_hourly.csv"))

HORIZON = 24 * 30
train, test = df.iloc[:-HORIZON], df.iloc[-HORIZON:]
feats = [c for c in df.columns if c != "MW"]

snaive = SeasonalNaive().fit(train[feats], train["MW"]).predict(test[feats])
gbm = GradientBoostingModel().fit(train[feats], train["MW"]).predict(test[feats])

y = test["MW"].to_numpy()
out = {
    "timestamps": [t.isoformat() for t in test.index],
    "actual": [round(v, 1) for v in y.tolist()],
    "seasonal_naive": [round(v, 1) for v in snaive.tolist()],
    "gbm": [round(v, 1) for v in gbm.tolist()],
    "metrics": {
        "seasonal_naive": {"mae": mae(y, snaive), "rmse": rmse(y, snaive), "mape": mape(y, snaive)},
        "gbm": {"mae": mae(y, gbm), "rmse": rmse(y, gbm), "mape": mape(y, gbm)},
    },
}
(ROOT / "docs" / "data.json").write_text(json.dumps(out))
print(f"wrote {len(y)} rows; GBM MAPE={out['metrics']['gbm']['mape']:.4f} "
      f"vs SeasonalNaive MAPE={out['metrics']['seasonal_naive']['mape']:.4f}")
