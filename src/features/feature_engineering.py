import pandas as pd


def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Add calendar/time features derived from the timestamp."""
    df = df.copy()
    ts = df[timestamp_col].dt

    df["hour"] = ts.hour
    df["day_of_week"] = ts.day_of_week
    df["month"] = ts.month
    df["is_weekend"] = (ts.day_of_week >= 5).astype(int)
    df["day_of_year"] = ts.day_of_year.astype(str)

    return df
