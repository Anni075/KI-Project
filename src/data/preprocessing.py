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


def compute_pv_surplus(df: pd.DataFrame) -> pd.DataFrame:
    """Add a pv_surplus column: Solarproduktion minus Hausverbrauch."""
    df = df.copy()
    df["pv_surplus"] = df["Solarproduktion"] - df["Hausverbrauch"]
    return df


def merge_features(
    pv_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    irr_df: pd.DataFrame,
    local_tz: str = "Europe/Berlin",
) -> pd.DataFrame:
    """
    Merge PV (15-min, local naive time), weather (hourly, local naive time)
    and irradiance (15-min, UTC-aware) into a single DataFrame.

    Weather is matched via backward fill (last known hourly value).
    Irradiance is matched to the nearest 15-min slot (≤15 min tolerance).
    """
    base = pv_df.sort_values("timestamp").reset_index(drop=True)

    # Weather: hourly, already in local naive time after load_weather_data()
    w_cols = ["timestamp", "temperature_2m", "cloud_cover", "cloud_cover_low",
              "relative_humidity_2m"]
    w = weather_df[w_cols].sort_values("timestamp").reset_index(drop=True)
    base = pd.merge_asof(base, w, on="timestamp",
                         direction="backward", tolerance=pd.Timedelta("1h"))

    # Irradiance: UTC-aware timestamps → convert to local naive time
    irr = irr_df[["timestamp", "ghi_cloudy_sky", "ghi_clear_sky"]].copy()
    irr["timestamp"] = (
        irr["timestamp"].dt.tz_convert(local_tz).dt.tz_localize(None)
    )
    irr = irr.sort_values("timestamp").reset_index(drop=True)
    base = pd.merge_asof(base, irr, on="timestamp",
                         direction="nearest", tolerance=pd.Timedelta("15min"))

    return base
