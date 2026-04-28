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
        lines = pred_path.read_text().splitlines()
        assert "prediction" in lines[0].lower()
        assert len(lines) > 1

    def test_unknown_subcommand_returns_nonzero(self):
        proc = _run(["wat"])
        assert proc.returncode != 0
