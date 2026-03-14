import numpy as np
import pandas as pd


def add_time_features(
    df: pd.DataFrame,
    target: str = "Solarproduktion",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """add phase-shifted cosine time features derived from the timestamp.

    All peaks are data-driven (no hardcoded constants).
    """
    df = df.copy()
    ts = df[timestamp_col]
    hour      = ts.dt.hour
    minute    = ts.dt.minute
    month     = ts.dt.month
    dayofyear = ts.dt.dayofyear

    # Stunden-Kosinus (Integer-Stunde, 24 Slots/Tag; window=3 ≈ 12 % von T=24)
    h_peak = df.groupby(hour)[target].median().rolling(3, center=True, min_periods=1).median().idxmax()
    df["hour_cos_shifted"] = np.cos(2 * np.pi * (hour - h_peak) / 24)

    # 15-min-Intervall-Kosinus (96 Slots/Tag; window=5 ≈ 5 % von T=96)
    interval = hour * 4 + minute // 15
    i_peak = df.groupby(interval)[target].median().rolling(5, center=True, min_periods=1).median().idxmax()
    df["interval_cos_shifted"] = np.cos(2 * np.pi * (interval - i_peak) / 96)

    # Monats-Kosinus (Periode 12; window=3 ≈ 25 % von T=12)
    m_peak = df.groupby(month)[target].mean().rolling(3, center=True, min_periods=1).median().idxmax()
    df["month_cos_shifted"] = np.cos(2 * np.pi * (month - m_peak) / 12)

    # Jahrestag-Kosinus (Periode 365; window=14 ≈ 4 % von T=365)
    d_peak = df.groupby(dayofyear)[target].mean().rolling(14, center=True, min_periods=1).median().idxmax()
    df["doy_cos_shifted"] = np.cos(2 * np.pi * (dayofyear - d_peak) / 365)

    return df


def add_irradiance_features(df: pd.DataFrame) -> pd.DataFrame:
    """add clear-sky index derived from ghi columns."""
    df = df.copy()
    ghi_clear = df["ghi_clear_sky"].replace(0, np.nan)
    df["clear_sky_index"] = (df["ghi_cloudy_sky"] / ghi_clear).clip(0, 1).fillna(0)
    return df


def add_lag_features(df: pd.DataFrame, target: str = "Solarproduktion", lags: list[int] | None = None) -> pd.DataFrame:
    """add lag features for the target column."""
    if lags is None:
        lags = [96]
    df = df.copy()
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)
    return df
