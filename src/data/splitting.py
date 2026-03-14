import pandas as pd

_SEASON_MAP = {12: "Winter", 1: "Winter", 2: "Winter",
               3: "Spring", 4: "Spring", 5: "Spring",
               6: "Summer", 7: "Summer", 8: "Summer",
               9: "Autumn", 10: "Autumn", 11: "Autumn"}

def time_series_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronologischer Train / Val / Test Split (kein Shuffling).

    Warum kein zufälliger Split?
    Bei Zeitreihen würde ein zufälliger Split Data Leakage verursachen:
    das Modell würde "zukünftige" Werte als Features sehen.
    Daher wird streng chronologisch aufgeteilt.

    Parameters
    ----------
    df           : DataFrame sortiert nach timestamp (oder wird hier sortiert)
    val_frac     : Anteil Validierungsdaten (default 15%)
    test_frac    : Anteil Testdaten (default 15%)
    timestamp_col: Name der Zeitstempel-Spalte

    Returns
    -------
    (train, val, test) als separate DataFrames
    """
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(df)
    val_start  = int(n * (1 - val_frac - test_frac))
    test_start = int(n * (1 - test_frac))

    train = df.iloc[:val_start].copy()
    val   = df.iloc[val_start:test_start].copy()
    test  = df.iloc[test_start:].copy()

    return train, val, test


def split_summary(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                  timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Gibt eine übersichtliche Tabelle mit den Split-Eigenschaften zurück."""
    rows = []
    for name, subset in [("train", train), ("val", val), ("test", test)]:
        rows.append({
            "split":  name,
            "von":    subset[timestamp_col].min().date(),
            "bis":    subset[timestamp_col].max().date(),
            "n":      len(subset),
            "anteil": f"{len(subset) / (len(train) + len(val) + len(test)):.1%}",
        })
    return pd.DataFrame(rows).set_index("split")
