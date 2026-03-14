from pathlib import Path

import pandas as pd

# Project root = two levels up from this file (src/data/preprocessing.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_WEATHER_RENAME = {
    "time":                       "timestamp",
    "temperature_2m (°C)":        "temperature_2m",
    "cloud_cover (%)":            "cloud_cover",
    "wind_speed_10m (km/h)":      "wind_speed_10m",
    "precipitation (mm)":         "precipitation",
    "weather_code (wmo code)":    "weather_code",
    "relative_humidity_2m (%)":   "relative_humidity_2m",
    "pressure_msl (hPa)":         "pressure_msl",
    "cloud_cover_low (%)":        "cloud_cover_low",
    "cloud_cover_mid (%)":        "cloud_cover_mid",
    "cloud_cover_high (%)":       "cloud_cover_high",
    "wind_gusts_10m (km/h)":      "wind_gusts_10m",
}


def load_pv_data(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        path = _PROJECT_ROOT / "data" / "raw" / "pv_data.csv"
    df = pd.read_csv(path, sep=";", parse_dates=["timestamp"])
    return df


def load_weather_data(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        path = _PROJECT_ROOT / "data" / "raw" / "weather.csv"
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.rename(columns=_WEATHER_RENAME)
    return df


def load_irradiance_data(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        path = _PROJECT_ROOT / "data" / "raw" / "irradiance_anonymized.csv"
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["dt_iso"], utc=True)
    df = df.drop(columns=["dt", "dt_iso", "timezone", "city_name", "lat", "lon"])
    return df


def clip_to_common_range(
    pv_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    irr_df: pd.DataFrame,
    local_tz: str = "Europe/Berlin",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Kürzt alle drei Rohdaten-DataFrames auf den gemeinsamen Zeitraum.

    PV und Wetter liegen als lokale naive Zeit vor, Einstrahlung als
    UTC-bewusster Zeitstempel. Der Schnitt wird in lokaler Zeit berechnet;
    für den Irradiance-Filter wird der Bereich zurück nach UTC konvertiert.
    """
    irr_local = irr_df["timestamp"].dt.tz_convert(local_tz).dt.tz_localize(None)

    start = max(pv_df["timestamp"].min(), weather_df["timestamp"].min(), irr_local.min())
    end   = min(pv_df["timestamp"].max(), weather_df["timestamp"].max(), irr_local.max())

    pv_out = pv_df[
        (pv_df["timestamp"] >= start) & (pv_df["timestamp"] <= end)
    ].reset_index(drop=True)
    w_out = weather_df[
        (weather_df["timestamp"] >= start) & (weather_df["timestamp"] <= end)
    ].reset_index(drop=True)

    start_utc = pd.Timestamp(start).tz_localize(local_tz).tz_convert("UTC")
    end_utc   = pd.Timestamp(end).tz_localize(local_tz).tz_convert("UTC")
    irr_out = irr_df[
        (irr_df["timestamp"] >= start_utc) & (irr_df["timestamp"] <= end_utc)
    ].reset_index(drop=True)

    return pv_out, w_out, irr_out


def load_processed_data(
    path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Lädt aufbereitete Daten aus data/processed/.

    split: "all"   → features.csv  (vollständiger Datensatz ohne Lag-Features)


           "val"   → val.csv
           "test"  → test.csv
    """
    fname = "features.csv"
    if path is None:
        path = _PROJECT_ROOT / "data" / "processed" / fname
    return pd.read_csv(path, parse_dates=["timestamp"])
