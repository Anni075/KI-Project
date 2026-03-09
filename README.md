# KI-Project

## Thema

Viertelstündliche Day-Ahead-Prognose der PV-Produktion mittels Machine Learning – zweistufiger Ansatz mit Grundlastabzug zur Bestimmung verfügbarer Einspeiseenergie.

> **Stufe 1:** ML-Modell prognostiziert die Solarproduktion [kW] in 15-Minuten-Schritten für den nächsten Tag.
> **Stufe 2:** Abzug einer datengetriebenen Grundlastschätzung (10. Perzentil des Hausverbrauchs nach Stunde × Wochentag) ergibt die verfügbare Überschussenergie zur Einspeisung.


## Forschungsfrage

In welchem Maße können Machine-Learning-Modelle die viertelstündliche PV-Produktion einer Photovoltaikanlage unter Nutzung von Wetter- und Einstrahlungsdaten vorhersagen?


## Scharfe Frage (Modelloutput)

- Wie hoch ist die PV-Produktion je 15 Minuten am nächsten Tag?
- Wie viel Energie steht nach Abzug der Grundlast zur Einspeisung zur Verfügung?


## Hypothesen

### H1
Hohe PV-Produktion führt nicht zwangsläufig zu verwertbarem Überschuss, da die Grundlast (dauerhafter Mindestverbrauch) diesen reduziert.

### H2
Modelle mit Wetter- und Einstrahlungsdaten prognostizieren die viertelstündliche PV-Produktion genauer als naive Prognosemethoden (z. B. „letzter gemessener Wert" oder historischer Durchschnitt).

### H3
Die Berücksichtigung zeitlicher Merkmale wie Uhrzeit und Jahreszeit verbessert die Prognose der viertelstündlichen PV-Produktion.

### H4
Nichtlineare Machine-Learning-Modelle sind besser geeignet zur Prognose der viertelstündlichen PV-Produktion als lineare Regressionsmodelle.


## Setup

```bash
uv sync                   # Abhängigkeiten installieren (Python ≥ 3.13)
uv run jupyter notebook   # Notebooks starten
```

Notebooks in dieser Reihenfolge ausführen: `01` → `02` → `03a` / `03b` → `03` → `04` → `04b` → `05`
