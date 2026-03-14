# KI-Project
## Thema
Viertelstündliche Day-Ahead-Prognose der PV-Produktion mittels Machine Learning.

* **Stufe 0:** Naive Baseline - [Notebook](notebooks/03_naive_baseline.ipynb)
* **Stufe 1:** Lineare Regression - [Notebook](notebooks/05_linear_regression.ipynb)
* **Stufe 2:** Random Forest - [Notebook](notebooks/06_random_forest.ipynb)

## Forschungsfrage
Wie genau können Machine-Learning-Modelle die viertelstündliche PV-Produktion anhand von Wetter- und Einstrahlungsdaten für den Folgetag vorhersagen?

## Scharfe Frage (Modelloutput)
Wie hoch ist die PV-Produktion je 15 Minuten am nächsten Tag in Watt?

## Hypothesen
| #  | Hypothese                                                                                                                    | Typ                    | Richtung  | H₀                                                                                            | H₁                                                            | Testverfahren                                   |
|----|------------------------------------------------------------------------------------------------------------------------------|------------------------|-----------|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------|-------------------------------------------------|
| H1 | Modelle mit Wetter- und Einstrahlungsdaten prognostizieren die PV-Produktion genauer als ein naiver Ansatz.                  | Unterschiedshypothese  | Gerichtet | MAE(ML) ≥ MAE(naiv)                                                                           | MAE(ML) < MAE(naiv)                                           | Paired t-Test / Wilcoxon / Diebold-Mariano-Test |
| H2 | Die Berücksichtigung zeitlicher Merkmale (Uhrzeit, Jahreszeit) verbessert die Prognose der viertelstündlichen PV-Produktion. | Unterschiedshypothese  | Gerichtet | MAE(mit Zeitfeatures) ≥ MAE(ohne)                                                             | MAE(mit Zeitfeatures) < MAE(ohne)                             | Paired t-Test / Wilcoxon auf Fehlerdifferenzen  |
| H3 | Die Prognosegenauigkeit korreliert negativ mit dem Bewölkungsgrad – an stark bewölkten Tagen sinkt die Vorhersagequalität.   | Zusammenhangshypothese | Gerichtet | Es besteht kein (oder ein positiver) Zusammenhang zwischen Bewölkungsgrad und Prognosefehler. | Je höher der Bewölkungsgrad, desto größer der Prognosefehler. | Korrelationsanalyse (Pearson / Spearman)        |
| H4 | Nichtlineare ML-Modelle sind besser geeignet zur Prognose als lineare Regressionsmodelle.                                    | Unterschiedshypothese  | Gerichtet | MAE(nichtlinear) ≥ MAE(linear)                                                                | MAE(nichtlinear) < MAE(linear)                                | Paired t-Test / Wilcoxon / Diebold-Mariano-Test |

## Setup
```bash
uv sync
```
