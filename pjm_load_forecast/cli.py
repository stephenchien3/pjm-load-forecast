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
