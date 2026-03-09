# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

15-minute day-ahead PV production forecasting for a single installation. Target variable: `Solarproduktion` [kW]. A post-processing step subtracts an estimated base load (`Hausverbrauch` 10th percentile by hour × weekday) to derive available surplus energy.

**Hypotheses being tested:** H2 (weather/irradiance > naive), H3 (time features help), H4 (nonlinear > linear).

## Commands

```bash
# Run notebooks
uv run jupyter notebook

# Run a Python script / ad-hoc check
uv run python -c "..."

# Add a dependency
uv add <package>
```

## Architecture

### Data flow

```
data/raw/          src/data/           src/features/         src/models/
pv_data.csv    →   load_pv_data()  \
weather.csv    →   load_weather_data() → merge_features() → add_time_features()
irradiance_    →   load_irradiance_      (merge_asof)       add_irradiance_features()  →  train/predict
anonymized.csv     data()                                   add_lag_features()
```

- **PV data** is 15-min, local naive time (no timezone). Weather is hourly; irradiance is 15-min UTC-aware (`dt_iso`). `merge_features()` handles timezone conversion (`Europe/Berlin`) and forward-filling.
- `compute_pv_surplus()` in `preprocessing.py` exists as a utility (computes `Solarproduktion − Hausverbrauch`) but is **not** used in the training pipeline — it's for the Grundlast analysis notebook only.

### Key constants (set at the top of every notebook)

```python
TARGET      = "Solarproduktion"
P_NOM       = 13_500.0          # installed capacity [kWp], used for nRMSE/nMAE
FEATURES_A  = ["ghi_cloudy_sky", "clear_sky_index", "hour_sin", "hour_cos",
                "day_sin", "day_cos", "temperature_2m", "cloud_cover_low"]
FEATURES_B  = FEATURES_A + ["Solarproduktion_lag_96"]
```

### Splits

`time_series_split()` does a strict chronological 70 / 15 / 15 split — no shuffling. Val set is summer/autumn-heavy; test set is winter/spring-heavy. Always call `.dropna(subset=features + [TARGET])` after adding lag features to remove NaNs at split boundaries.

### Evaluation

`evaluate(y_true, y_pred, P_NOM, y_ref=pers_val)` returns 11 metrics including `skill_mae` (vs. persistence), `nrmse`, `mbe`, `ramp_mae`. `evaluate_by_season()` repeats per season. Results are saved as JSON to `results/<notebook_name>/`.

### Notebooks

| Notebook | Purpose |
|---|---|
| `01_exploration` | EDA |
| `02_feature_engineering` | Feature exploration |
| `03a/03b_naive_*` | Fit & save naive baselines (run before 03) |
| `03_naive_analysis` | Load results from 03a/03b and compare |
| `04_linear_regression` | Model A (no lag) vs Model B (+ lag₉₆), saves JSON |
| `04b_feature_importance` | Correlation, normed coefficients, permutation importance, ablation |
| `05_grundlast_analyse` | Base load analysis; saves `results/05_grundlast/grundlast_lookup.json` |

### Data files

| File | Timestamp col | Granularity | Notes |
|---|---|---|---|
| `pv_data.csv` | `timestamp` | 15 min | Semicolon-separated |
| `weather.csv` | `time` (→ renamed `timestamp`) | hourly | Column names include units, stripped on load |
| `irradiance_anonymized.csv` | `dt_iso` (UTC+00:00) | 15 min | Converted to `Europe/Berlin` naive in `merge_features()` |
