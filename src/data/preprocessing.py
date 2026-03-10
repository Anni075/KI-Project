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


def merge_features(
    pv_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    irr_df: pd.DataFrame,
    local_tz: str = "Europe/Berlin",
) -> pd.DataFrame:
    """
    Zusammenführen von PV- (15-min, lokale naive Zeit), Wetter- (stündlich,
    lokale naive Zeit) und Einstrahlungsdaten (15-min, UTC-bewusst) zu einem
    einzigen DataFrame.

    Wetter: Die stündlichen Werte werden per linearer Interpolation auf ein
    15-Minuten-Raster hochskaliert, um einen glatten zeitlichen Verlauf
    sicherzustellen (kein Stufensprung an Stundengrenzen).
    Merge über merge_asof mit Toleranz ≤ 15 min.

    Einstrahlung: UTC-Zeitstempel → lokale naive Zeit, dann merge_asof auf den
    nächsten 15-Minuten-Slot (Toleranz ≤ 15 min).
    """
    base = pv_df.sort_values("timestamp").reset_index(drop=True)

    # Wetterdaten: stündlich, bereits in lokaler naiver Zeit (nach load_weather_data()).
    # Durch lineare Interpolation auf ein 15-Minuten-Raster hochskalieren,
    # damit keine Sprünge an Stundengrenzen entstehen, sondern ein glatter Verlauf.
    w_cols = ["timestamp", "temperature_2m", "cloud_cover", "cloud_cover_low",
              "relative_humidity_2m"]
    w = (
        weather_df[w_cols]
        .sort_values("timestamp")
        .set_index("timestamp")
        .resample("15min")
        .interpolate(method="linear")
        .reset_index()
    )
    base = pd.merge_asof(base, w, on="timestamp",
                         direction="nearest", tolerance=pd.Timedelta("15min"))

    # Irradiance: UTC-aware timestamps → convert to local naive time
    irr = irr_df[["timestamp", "ghi_cloudy_sky", "ghi_clear_sky"]].copy()
    irr["timestamp"] = (
        irr["timestamp"].dt.tz_convert(local_tz).dt.tz_localize(None)
    )
    irr = irr.sort_values("timestamp").reset_index(drop=True)
    base = pd.merge_asof(base, irr, on="timestamp",
                         direction="nearest", tolerance=pd.Timedelta("15min"))

    return base
