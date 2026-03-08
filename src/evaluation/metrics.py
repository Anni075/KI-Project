import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(r2_score(y_true, y_pred))


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict:
    mask = y_true.notna() & y_pred.notna()
    return {
        "rmse": rmse(y_true[mask], y_pred[mask]),
        "mae": mae(y_true[mask], y_pred[mask]),
        "r2": r2(y_true[mask], y_pred[mask]),
    }
