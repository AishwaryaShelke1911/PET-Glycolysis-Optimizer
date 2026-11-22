"""
optimize.py
Given a chosen catalyst (and optional constraints), searches over a grid of
temperature / time / dosage / EG ratio combinations to recommend the
conditions predicted to maximize BHET yield.

IMPORTANT - extrapolation caveat:
Tree-based models (Random Forest / XGBoost) do not extrapolate smoothly
outside the range of conditions actually seen in training. If the grid
search range is wider than what was tested for a given catalyst in the
literature, many grid points will fall into the same decision-tree leaf
and return identical, overconfident predictions. To avoid this, the grid
is automatically restricted to the range actually observed for the chosen
catalyst (with a small margin), unless the caller explicitly overrides it.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

from predict import load_model
from preprocessing import build_features

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "glycolysis_clean.csv"


def _get_catalyst_observed_ranges(catalyst: str, margin: float = 0.15):
    """
    Look up the min/max of each numeric feature actually observed for this
    catalyst in the training data, expanded by a small margin. Falls back
    to the full dataset's range if the catalyst has too few rows.
    """
    defaults = {
        "Temp_C": (170, 230),
        "Time_min": (30, 180),
        "Catalyst_Dosage_wt%": (0.5, 5.0),
        "EG_PET_Ratio": (2, 8),
    }

    if not PROCESSED_PATH.exists():
        return defaults

    df = pd.read_csv(PROCESSED_PATH)
    subset = df[df["Catalyst"] == catalyst]

    if len(subset) < 3:
        subset = df  # not enough catalyst-specific data, use whole dataset range

    physical_minimums = {
        "Temp_C": 100,
        "Time_min": 5,
        "Catalyst_Dosage_wt%": 0.1,
        "EG_PET_Ratio": 0.5,
    }

    ranges = {}
    for col in ["Temp_C", "Time_min", "Catalyst_Dosage_wt%", "EG_PET_Ratio"]:
        col_data = subset[col].dropna()
        if len(col_data) == 0:
            ranges[col] = defaults[col]
            continue
        # Use 5th/95th percentile rather than raw min/max so a single
        # extreme outlier row (e.g. an uncatalyzed reaction reported as
        # taking thousands of minutes) doesn't blow out the search grid.
        lo, hi = col_data.quantile(0.05), col_data.quantile(0.95)
        if lo == hi:
            lo, hi = col_data.min(), col_data.max()
        span = max(hi - lo, 1e-6)
        lo_bound = max(physical_minimums[col], lo - margin * span)
        hi_bound = hi + margin * span
        ranges[col] = (lo_bound, hi_bound)

    return ranges


def recommend_conditions(
    catalyst: str,
    temp_range=None,
    time_range=None,
    dosage_range=None,
    eg_ratio_range=None,
    n_points: int = 8,
    top_n: int = 5,
    add_jitter: bool = True,
):
    """
    Grid-search over reaction condition combinations for a given catalyst and
    return the top_n combinations predicted to give the highest BHET yield.

    By default, ranges are auto-restricted to conditions actually observed
    for this catalyst in the literature dataset (plus a small margin) to
    avoid extrapolating into untested territory. Pass explicit *_range
    tuples (min, max) to override.

    n_points controls grid resolution per dimension (n_points^4 total combos).
    add_jitter adds a tiny deterministic tiebreaker so visually-identical
    predictions don't all collapse into duplicate rows.
    """
    model = load_model()
    observed = _get_catalyst_observed_ranges(catalyst)

    temp_range = temp_range or observed["Temp_C"]
    time_range = time_range or observed["Time_min"]
    dosage_range = dosage_range or observed["Catalyst_Dosage_wt%"]
    eg_ratio_range = eg_ratio_range or observed["EG_PET_Ratio"]

    temps = np.linspace(*temp_range, n_points)
    times = np.linspace(*time_range, n_points)
    dosages = np.linspace(*dosage_range, n_points)
    eg_ratios = np.linspace(*eg_ratio_range, n_points)

    combos = list(product(temps, times, dosages, eg_ratios))
    grid_df = pd.DataFrame(combos, columns=["Temp_C", "Time_min", "Catalyst_Dosage_wt%", "EG_PET_Ratio"])
    grid_df["Catalyst"] = catalyst

    X, _ = build_features(grid_df, fit=False)
    preds = model.predict(X)

    if add_jitter:
        rng = np.random.default_rng(42)
        preds = preds + rng.normal(0, 1e-3, size=len(preds))

    grid_df["Predicted_BHET_Yield_pct"] = preds

    # Drop near-duplicate rows (same predicted yield to 2 decimal places)
    # so the table shows variety rather than repeats from flat tree leaves.
    grid_df["_yield_rounded"] = grid_df["Predicted_BHET_Yield_pct"].round(2)
    grid_df = grid_df.drop_duplicates(subset="_yield_rounded").drop(columns="_yield_rounded")

    top_results = grid_df.sort_values("Predicted_BHET_Yield_pct", ascending=False).head(top_n)
    return top_results.reset_index(drop=True)


if __name__ == "__main__":
    results = recommend_conditions(catalyst="Zinc acetate")
    print("Top recommended conditions for Zinc acetate:\n")
    print(results.to_string(index=False))
