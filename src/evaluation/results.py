from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_results(paths: list[Path | str]) -> dict[str, dict]:
    """Load result JSONs. Returns {model_name: result_dict}."""
    out = {}
    for p in paths:
        with open(p) as f:
            r = json.load(f)
        out[r["model"]] = r
    return out


def get_split_metrics(results: dict[str, dict], split: str = "val") -> pd.DataFrame:
    """
    Extract a DataFrame with one row per model, one column per metric,
    for the given split. Useful for comparison tables and bar charts.
    """
    rows = {
        model_name: r["splits"][split]
        for model_name, r in results.items()
        if split in r.get("splits", {})
    }
    return pd.DataFrame(rows).T
