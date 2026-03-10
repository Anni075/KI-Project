# KI-Project

## Thema

Viertelstündliche Day-Ahead-Prognose der PV-Produktion mittels Machine Learning – zweistufiger Ansatz mit Grundlastabzug zur Bestimmung verfügbarer Einspeiseenergie.

> **Stufe 1:** ML-Modell prognostiziert die Solarproduktion [kW] in 15-Minuten-Schritten für den nächsten Tag.
> **Stufe 2:** Abzug einer datengetriebenen Grundlastschätzung (10. Perzentil des Hausverbrauchs nach Stunde × Wochentag) ergibt die verfügbare Überschussenergie zur Einspeisung.


## Forschungsfrage

In welchem Maße können Machine-Learning-Modelle die viertelstündliche PV-Produktion einer Photovoltaikanlage unter Nutzung von Wetter- und Einstrahlungsdaten vorhersagen?


## Scharfe Frage (Modelloutput)

- Wie hoch ist die PV-Produktion je 15 Minuten am nächsten Tag?


## Hypothesen


Hypothesen folie · MD
Copy

# Hypothesen – Übersicht

| # | Hypothese | Typ | Richtung | H₀ | H₁ | Testverfahren |
|---|-----------|-----|----------|----|----|---------------|
| H1 | Die Prognosegenauigkeit korreliert negativ mit dem Bewölkungsgrad – an stark bewölkten Tagen sinkt die Vorhersagequalität. | Zusammenhangshypothese | Gerichtet | Es besteht kein (oder ein positiver) Zusammenhang zwischen Bewölkungsgrad und Prognosefehler. | Je höher der Bewölkungsgrad, desto größer der Prognosefehler. | Korrelationsanalyse (Pearson / Spearman) |
| H2 | Modelle mit Wetter- und Einstrahlungsdaten prognostizieren die PV-Produktion genauer als naive Methoden (Durchschnitt, Vortag). | Unterschiedshypothese | Gerichtet | MAE(ML) ≥ MAE(naiv) | MAE(ML) < MAE(naiv) | Paired t-Test / Wilcoxon / Diebold-Mariano-Test |
| H3 | Die Berücksichtigung zeitlicher Merkmale (Uhrzeit, Jahreszeit) verbessert die Prognose der viertelstündlichen PV-Produktion. | Unterschiedshypothese | Gerichtet | MAE(mit Zeitfeatures) ≥ MAE(ohne) | MAE(mit Zeitfeatures) < MAE(ohne) | Paired t-Test / Wilcoxon auf Fehlerdifferenzen |
| H4 | Nichtlineare ML-Modelle sind besser geeignet zur Prognose als lineare Regressionsmodelle. | Unterschiedshypothese | Gerichtet | MAE(nichtlinear) ≥ MAE(linear) | MAE(nichtlinear) < MAE(linear) | Paired t-Test / Wilcoxon / Diebold-Mariano-Test |

## Setup

```bash
uv sync                   # Abhängigkeiten installieren (Python ≥ 3.13)
uv run jupyter notebook   # Notebooks starten
```

Notebooks in dieser Reihenfolge ausführen: `01` → `02` → `03a` / `03b` → `03` → `04` → `04b` → `05`
