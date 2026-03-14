# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

15-minute day-ahead PV production forecasting for a single installation. Target variable: `Solarproduktion` [W]. A post-processing step subtracts an estimated base load (`Hausverbrauch` 10th percentile by hour × weekday) to derive available surplus energy.

**Hypotheses being tested:** H1 (weather/irradiance > naive), H2 (time features help), H3 (cloud coverage vs. forecast error), H4 (nonlinear > linear).

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

### Key constants

```python
TARGET = "Solarproduktion"
P_NOM  = 13_500.0  # installed capacity in W (= 13.5 kWp), used for nRMSE/nMAE
```

### Feature sets — `src/features/feature_sets.py`

Single source of truth for all feature sets (`BASE`, `BASE_WITH_TIME`, `BASE_WITH_TIME_AND_LAG`). In notebooks: `FEATURES_A = FEATURE_SETS["BASE"].features`.

### Splits

`time_series_split()` does a strict chronological 70 / 15 / 15 split — no shuffling. Val set is summer/autumn-heavy; test set is winter/spring-heavy. Always call `.dropna(subset=features + [TARGET])` after adding lag features to remove NaNs at split boundaries.

### Evaluation & pipeline — `src/models/pipeline.py`

`run_pipeline(model_name, predict_fn, train, val, features, feature_set_key, params, results_dir)` evaluates on train+val, saves JSON without test block.

`evaluate_on_test(result, predict_fn, test, features, results_dir)` appends test metrics and re-saves. Call once, deliberately, via `RUN_TEST = True` at the top of the notebook.

`predict_fn(df, features) -> pd.Series` with DatetimeIndex. For sklearn: `lambda df, feats: pd.Series(model.predict(df.set_index("timestamp")[feats]), index=...)`. For naive: `lambda df, _: predict_fn(means, df).set_axis(...)`.

### Result JSON format

```json
{
  "model": "random_forest_B",
  "feature_set": "B",
  "features": ["ghi_cloudy_sky", "..."],
  "params": {},
  "splits": {
    "train": { "rmse": 0.0, "mae": 0.0, "mbe": 0.0, "nrmse": 0.0, "nmae": 0.0,
               "nmbe": 0.0, "r2": 0.0, "mape_daytime": 0.0,
               "skill_mae": 0.0, "skill_rmse": 0.0, "ramp_mae": 0.0 },
    "val":   { ... },
    "test":  { ... }
  },
  "by_season": {
    "train": { "Winter": {...}, "Spring": {...}, "Summer": {...}, "Autumn": {...} },
    "val":   { ... },
    "test":  { ... }
  }
}
```

`test` block only present after `evaluate_on_test()` is called.

### Loading results — `src/evaluation/results.py`

Helpers for hypothesis notebooks: `load_results()` and `get_split_metrics()`.

### Notebooks

| Notebook                                          | Purpose                                                       |
|---------------------------------------------------|---------------------------------------------------------------|
| `notebooks/00_data_preparation`                   | Data loading, merging, quality checks, save `data/processed/` |
| `notebooks/01_univariate_analyse`                 | EDA: descriptive stats, temporal patterns, example week       |
| `notebooks/02_outliers_analysis`                  | Outlier detection                                             |
| `notebooks/02_splitting`                          | Chronological train/val/test split (70/15/15)                 |
| `notebooks/04_feature_engineering`                | Feature exploration and selection                             |
| `models/00_naive_baseline`                        | Seasonal climatology baseline, saves JSON via `run_pipeline`  |
| `models/01_linear_regression`                     | Linear A (no time) vs B (+ time features), saves JSON         |
| `models/02_random_forest`                         | Random Forest A/B, feature importance, saves JSON             |
| `hypothesen/H01_naive_vs_weather`                 | H1: loads JSONs, paired test naive vs. weather-driven         |
| `hypothesen/H02_temporal_features_impact`         | H2: loads JSONs, tests time feature impact                    |
| `hypothesen/H03_cloud_coverage_vs_forecast_error` | H3: loads JSONs, cloud coverage vs. error                     |
| `hypothesen/H04_linear_vs_nonlinear_models`       | H4: loads JSONs, linear vs. nonlinear comparison              |

### Data files

| File                        | Timestamp col                  | Granularity | Notes                                                    |
|-----------------------------|--------------------------------|-------------|----------------------------------------------------------|
| `pv_data.csv`               | `timestamp`                    | 15 min      | Semicolon-separated                                      |
| `weather.csv`               | `time` (→ renamed `timestamp`) | hourly      | Column names include units, stripped on load             |
| `irradiance_anonymized.csv` | `dt_iso` (UTC+00:00)           | 15 min      | Converted to `Europe/Berlin` naive in `merge_features()` |
