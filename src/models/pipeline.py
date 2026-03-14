from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd

from src.evaluation.metrics import evaluate, evaluate_by_season

P_NOM = 13_500.0
TARGET = "Solarproduktion"


def _evaluate_split(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_ref: pd.Series | None = None,
) -> tuple[dict, dict]:
    metrics = evaluate(y_true, y_pred, P_NOM, y_ref=y_ref)
    season_df = evaluate_by_season(y_true, y_pred, P_NOM, y_ref=y_ref)
    return metrics, season_df.to_dict(orient="index")


def _save(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"Saved: {path}")


def _save_predictions(y_true: pd.Series, y_pred: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).rename_axis("timestamp").to_csv(path)
    print(f"Saved predictions: {path}")


def run_pipeline(
    model_name: str,
    predict_fn: Callable,
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    feature_set_key: str | None = None,
    params: dict | None = None,
    results_dir: Path | str | None = None,
    ref_predict_fn: Callable | None = None,
) -> dict:
    """
    Evaluate a model on train and val. Saves JSON without test block.

    predict_fn(df, features) -> pd.Series with DatetimeIndex aligned to df.
    For sklearn models: lambda df, feats: pd.Series(model.predict(df.set_index("timestamp")[feats]), index=...)
    For naive models: lambda df, _: predict_fn(fitted_params, df).set_axis(df.set_index("timestamp").index)

    ref_predict_fn: optional, same signature as predict_fn; produces the skill-score baseline.
    If None, metrics.py falls back to day-ahead persistence internally.
    """
    y_train = train.set_index("timestamp")[TARGET]
    y_val = val.set_index("timestamp")[TARGET]

    pred_train = predict_fn(train, features)
    pred_val = predict_fn(val, features)

    y_ref_train = ref_predict_fn(train, features) if ref_predict_fn else None
    y_ref_val   = ref_predict_fn(val,   features) if ref_predict_fn else None

    train_metrics, train_season = _evaluate_split(y_train, pred_train, y_ref_train)
    val_metrics, val_season = _evaluate_split(y_val, pred_val, y_ref_val)

    result = {
        "model": model_name,
        "feature_set": feature_set_key,
        "features": features,
        "params": params or {},
        "splits": {"train": train_metrics, "val": val_metrics},
        "by_season": {"train": train_season, "val": val_season},
    }

    if results_dir is not None:
        base = Path(results_dir)
        _save(result, base / f"{model_name}.json")
        _save_predictions(y_train, pred_train, base / f"{model_name}_predictions_train.csv")
        _save_predictions(y_val,   pred_val,   base / f"{model_name}_predictions_val.csv")

    return result


def evaluate_on_test(
    result: dict,
    predict_fn: Callable,
    test: pd.DataFrame,
    features: list[str],
    results_dir: Path | str | None = None,
    ref_predict_fn: Callable | None = None,
) -> dict:
    """
    Append test metrics to result dict and re-save. Call once, deliberately.
    """
    y_test = test.set_index("timestamp")[TARGET]
    pred_test = predict_fn(test, features)

    y_ref_test = ref_predict_fn(test, features) if ref_predict_fn else None
    test_metrics, test_season = _evaluate_split(y_test, pred_test, y_ref_test)

    result["splits"]["test"] = test_metrics
    result["by_season"]["test"] = test_season

    if results_dir is not None:
        base = Path(results_dir)
        _save(result, base / f"{result['model']}.json")
        _save_predictions(y_test, pred_test, base / f"{result['model']}_predictions_test.csv")

    return result
