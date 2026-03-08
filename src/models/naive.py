import numpy as np
import pandas as pd


def persistence_forecast(series: pd.Series, lag: int = 96) -> pd.Series:
    """Naive persistence baseline: predict the value from `lag` steps ago (default 24h = 96 * 15min)."""
    return series.shift(lag)


def historical_mean_forecast(series: pd.Series, window: int = 96) -> pd.Series:
    """Rolling historical mean over the last `window` steps."""
    return series.shift(1).rolling(window=window, min_periods=1).mean()
