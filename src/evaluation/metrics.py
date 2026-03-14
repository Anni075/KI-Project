import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── Basismetriken ────────────────────────────────────────────────────────────

def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def mbe(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Bias Error – zeigt systematische Über-/Unterschätzung.

    Stärken:  Systemfehler sichtbar (>0 = Überschätzung, <0 = Unterschätzung).
    Schwächen: Kein Maß für Streuung – positive und negative Fehler heben sich auf.
    Anwendung: Kalibrierung des Modells.
    """
    return float((y_pred - y_true).mean())


def r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Bestimmtheitsmaß – Anteil der durch das Modell erklärten Varianz.

    Stärken:  Erklärte Varianz intuitiv interpretierbar (0–1).
    Schwächen: Wenig aussagekräftig bei Skalierungsunterschieden zwischen Splits.
    Anwendung: Feature- und Modellentwicklung.
    """
    return float(r2_score(y_true, y_pred))


# ── Normalisierte Metriken (Standard im PV-Forecasting) ─────────────────────

def nmae(y_true: pd.Series, y_pred: pd.Series, p_nom: float) -> float:
    """Normalized MAE – normalisiert auf installierte Nennleistung [kWp].

    Stärken:  Interpretierbar, robust gegenüber Ausreißern.
    Schwächen: Ignoriert Varianz – große und kleine Fehler gleich gewichtet.
    Anwendung: Baseline-Vergleich, Kostenrechnung.
    """
    return mae(y_true, y_pred) / p_nom


def nrmse(y_true: pd.Series, y_pred: pd.Series, p_nom: float) -> float:
    """Normalized RMSE – normalisiert auf installierte Nennleistung [kWp].

    Stärken:  Sensibel für Spitzenfehler (quadratische Gewichtung).
    Schwächen: Überempfindlich bei Outliern.
    Anwendung: Wolkenramps / Day-Ahead-Volatilität.
    """
    return rmse(y_true, y_pred) / p_nom


def nmbe(y_true: pd.Series, y_pred: pd.Series, y_true_mean: float | None = None) -> float:
    """Normalized MBE – normalisiert auf den Mittelwert der Messung."""
    ref = y_true_mean if y_true_mean is not None else float(y_true.mean())
    return mbe(y_true, y_pred) / ref if ref != 0 else float("nan")


# ── MAPE (nur für Tagesstunden sinnvoll) ─────────────────────────────────────

def mape_daytime(y_true: pd.Series, y_pred: pd.Series,
                 threshold: float = 0.01) -> float:
    """
    Mean Absolute Percentage Error – nur für Werte > threshold,
    um Division durch ~0 in Nacht-/Morgenstunden zu vermeiden.
    """
    mask = y_true > threshold
    if mask.sum() == 0:
        return float("nan")
    return float((((y_true[mask] - y_pred[mask]).abs() / y_true[mask])).mean())


# ── Skill Score gegenüber Persistenz-Baseline ────────────────────────────────

def persistence_forecast(y_true: pd.Series, horizon_steps: int = 96) -> pd.Series:
    """
    Naiver Day-Ahead Persistenz-Forecast:
    Prognose für Tag t = Messung von Tag t-1 (96 × 15-min-Schritte).
    """
    return y_true.shift(horizon_steps)


def skill_score(y_true: pd.Series, y_pred: pd.Series,
                y_ref: pd.Series, metric_fn=mae) -> float:
    """
    Skill Score SS = 1 - metric(model) / metric(baseline).
    SS > 0: Modell besser als Baseline; SS = 1: perfekt.
    """
    score_model = metric_fn(y_true, y_pred)
    score_ref   = metric_fn(y_true, y_ref)
    return float(1 - score_model / score_ref) if score_ref != 0 else float("nan")


# ── Ramp-Metriken (wichtig für Netz-/Speichersteuerung) ─────────────────────

def ramp_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    MAE der 15-min-Rampen (Delta zwischen aufeinanderfolgenden Zeitschritten).
    Bewertet, wie gut schnelle Leistungsänderungen vorhergesagt werden.
    """
    ramp_true = y_true.diff().dropna()
    ramp_pred = y_pred.diff().dropna()
    return mae(ramp_true, ramp_pred)


# ── Hauptfunktion ─────────────────────────────────────────────────────────────

def evaluate(
    y_true: pd.Series,
    y_pred: pd.Series,
    p_nom: float,
    y_ref: pd.Series | None = None,
    daytime_threshold: float = 0.01,
) -> dict:
    """
    Vollständige Metrik-Suite für Day-Ahead PV-Forecasting.

    Parameters
    ----------
    y_true : gemessene PV-Produktion [kW]
    y_pred : prognostizierte PV-Produktion [kW]
    p_nom  : installierte Nennleistung [kWp] – für Normierung
    y_ref  : Persistenz-Baseline (optional, wird sonst intern berechnet)
    daytime_threshold : Untergrenze für MAPE-Berechnung [kW]
    """
    mask = y_true.notna() & y_pred.notna()
    yt, yp = y_true[mask], y_pred[mask]

    if y_ref is None:
        y_ref = persistence_forecast(y_true)
    yr = y_ref[mask]
    ref_mask = yr.notna()

    metrics = {
        # Absolute Fehler
        "rmse":          rmse(yt, yp),
        "mae":           mae(yt, yp),
        "mbe":           mbe(yt, yp),
        # Normalisiert auf Nennleistung (Industriestandard)
        "nrmse":         nrmse(yt, yp, p_nom),
        "nmae":          nmae(yt, yp, p_nom),
        "nmbe":          nmbe(yt, yp),
        # Güte
        "r2":            r2(yt, yp),
        # Prozentualer Fehler (nur Tagesstunden)
        "mape_daytime":  mape_daytime(yt, yp, threshold=daytime_threshold),
        # Skill Score vs. Persistenz
        "skill_mae":     skill_score(yt[ref_mask], yp[ref_mask], yr[ref_mask], mae),
        "skill_rmse":    skill_score(yt[ref_mask], yp[ref_mask], yr[ref_mask], rmse),
        # Rampen
        "ramp_mae":      ramp_score(yt, yp),
    }
    return metrics


_SEASON_MAP = {12: "Winter", 1: "Winter", 2: "Winter",
               3: "Spring", 4: "Spring", 5: "Spring",
               6: "Summer", 7: "Summer", 8: "Summer",
               9: "Autumn", 10: "Autumn", 11: "Autumn"}

_SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]


def evaluate_by_season(
    y_true: pd.Series,
    y_pred: pd.Series,
    p_nom: float,
    y_ref: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Wertet evaluate() separat für jede Jahreszeit aus.

    y_true und y_pred müssen einen DatetimeIndex haben.
    Gibt einen DataFrame zurück (eine Zeile pro Saison).
    """
    seasons = y_true.index.month.map(_SEASON_MAP)
    rows = {}
    for season in _SEASON_ORDER:
        mask = seasons == season
        if mask.sum() == 0:
            continue
        yt_s = y_true[mask]
        yp_s = y_pred[mask]
        yr_s = y_ref[mask] if y_ref is not None else None
        rows[season] = evaluate(yt_s, yp_s, p_nom, y_ref=yr_s)
    return pd.DataFrame(rows).T
