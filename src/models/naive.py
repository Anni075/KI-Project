import pandas as pd

from src.data.splitting import _SEASON_MAP


def persistence_forecast(series: pd.Series, lag: int = 96) -> pd.Series:
    """
    Naiver Persistenz-Forecast: Prognose = Wert von vor `lag` Schritten.

    Bei 15-min-Daten und lag=96 entspricht das dem gleichen 15-min-Slot
    vom Vortag (Day-Ahead Persistenz).

    Wird auf der *gesamten* Serie berechnet – beim Auswerten nur den
    relevanten Zeitraum (val/test) herausschneiden.
    """
    return series.shift(lag)


def historical_mean_forecast(series: pd.Series, window: int = 96) -> pd.Series:
    """
    Rollender Mittelwert der letzten `window` Schritte (ohne aktuellen Wert).
    Nur rückwärtsschauend – kein Data Leakage.
    """
    return series.shift(1).rolling(window=window, min_periods=1).mean()


def fit_climatological_mean(
    train: pd.DataFrame,
    target_col: str,
    timestamp_col: str = "timestamp",
) -> pd.Series:
    """
    Berechnet den mittleren Tagesgang (pro Stunde) aus den Trainingsdaten.

    Returns
    -------
    pd.Series mit index = Stunde (0–23) und Wert = Mittelwert
    """
    return (
        train.groupby(train[timestamp_col].dt.hour)[target_col]
        .mean()
        .rename("clim_mean_by_hour")
    )


def predict_climatological_mean(
    hourly_means: pd.Series,
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.Series:
    """
    Wendet den gefitteten Stunden-Mittelwert auf einen DataFrame an.

    Parameters
    ----------
    hourly_means : Ausgabe von fit_climatological_mean()
    df           : DataFrame auf dem vorhergesagt wird (val oder test)

    Returns
    -------
    pd.Series mit gleichen Index wie df
    """
    return pd.Series(
        df[timestamp_col].dt.hour.map(hourly_means).values,
        index=df.index,
        name="clim_forecast",
    )


def fit_climatological_mean_by_season(
    train: pd.DataFrame,
    target_col: str,
    timestamp_col: str = "timestamp",
) -> pd.Series:
    """
    Berechnet den Stunden-Mittelwert konditioniert auf Jahreszeit (4×24 = 96 Werte).

    Returns
    -------
    pd.Series mit MultiIndex (season, hour)
    """
    seasons = train[timestamp_col].dt.month.map(_SEASON_MAP)
    hours   = train[timestamp_col].dt.hour
    return (
        train[target_col]
        .groupby([seasons, hours])
        .mean()
        .rename_axis(["season", "hour"])
        .rename("clim_seasonal_mean")
    )


def predict_climatological_mean_by_season(
    seasonal_means: pd.Series,
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.Series:
    """Wendet den saisonalen Stunden-Mittelwert auf einen DataFrame an."""
    keys = list(zip(
        df[timestamp_col].dt.month.map(_SEASON_MAP),
        df[timestamp_col].dt.hour,
    ))
    values = [seasonal_means.get((s, h), float("nan")) for s, h in keys]
    return pd.Series(values, index=df.index, name="clim_seasonal_forecast")


def fit_climatological_mean_by_month(
    train: pd.DataFrame,
    target_col: str,
    timestamp_col: str = "timestamp",
) -> pd.Series:
    """
    Berechnet den Stunden-Mittelwert konditioniert auf Kalendermonat (12×24 = 288 Werte).

    Returns
    -------
    pd.Series mit MultiIndex (month, hour)
    """
    months = train[timestamp_col].dt.month
    hours  = train[timestamp_col].dt.hour
    return (
        train[target_col]
        .groupby([months, hours])
        .mean()
        .rename_axis(["month", "hour"])
        .rename("clim_monthly_mean")
    )


def predict_climatological_mean_by_month(
    monthly_means: pd.Series,
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.Series:
    """Wendet den monatlichen Stunden-Mittelwert auf einen DataFrame an."""
    keys = list(zip(
        df[timestamp_col].dt.month,
        df[timestamp_col].dt.hour,
    ))
    values = [monthly_means.get((m, h), float("nan")) for m, h in keys]
    return pd.Series(values, index=df.index, name="clim_monthly_forecast")
