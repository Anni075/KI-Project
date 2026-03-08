import pandas as pd


def load_pv_data(path: str = "data/raw/pv_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", parse_dates=["timestamp"])
    return df


def load_weather_data(path: str = "data/raw/weather.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def load_irradiance_data(path: str = "data/raw/irradiance_anonymized.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def compute_pv_surplus(df: pd.DataFrame) -> pd.DataFrame:
    """Add a pv_surplus column: Solarproduktion minus Hausverbrauch."""
    df = df.copy()
    df["pv_surplus"] = df["Solarproduktion"] - df["Hausverbrauch"]
    return df
