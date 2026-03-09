import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Add calendar/time features derived from the timestamp."""
    df = df.copy()
    ts = df[timestamp_col]
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    # Cyclic encodings (linear regression cannot learn that hour 23 ≈ hour 0)
    df["day_of_year"] = ts.dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    df["day_sin"]  = np.sin(2 * np.pi * ts.dt.dayofyear / 365)
    df["day_cos"]  = np.cos(2 * np.pi * ts.dt.dayofyear / 365)
    return df


def add_irradiance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add clear-sky index: ratio of actual to potential irradiance under clear sky."""
    df = df.copy()
    df["clear_sky_index"] = (
        df["ghi_cloudy_sky"] / (df["ghi_clear_sky"] + 1e-6)
    ).clip(0, 1.5)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str = "Solarproduktion", lags: list[int] = None) -> pd.DataFrame:
    """Add lagged versions of the target column."""
    if lags is None:
        lags = [1, 4, 96]  # 15 min, 1 hour, 24 hours back
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df
