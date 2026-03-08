import pandas as pd


def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Add calendar/time features derived from the timestamp."""
    df = df.copy()
    ts = df[timestamp_col]
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str = "pv_surplus", lags: list[int] = None) -> pd.DataFrame:
    """Add lagged versions of the target column."""
    if lags is None:
        lags = [1, 4, 96]  # 15 min, 1 hour, 24 hours back
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df
