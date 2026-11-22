"""
predict.py
Loads the trained model and provides a prediction function with
honest uncertainty bands derived from Random Forest tree variance.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from preprocessing import build_features

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_model():
    return joblib.load(MODELS_DIR / "best_model.pkl")


def load_uncertainty_model():
    return joblib.load(MODELS_DIR / "uncertainty_model.pkl")


def predict_yield(catalyst: str, dosage_wt_pct: float, temp_c: float,
                  time_min: float, eg_pet_ratio: float):
    """
    Predict BHET yield (%) with an uncertainty band.

    Returns
    -------
    dict with:
      - predicted_yield: float (best estimate)
      - lower: float (lower bound of ~80% prediction interval)
      - upper: float (upper bound of ~80% prediction interval)
      - activation_energy_used: float (Ea used for this catalyst in kJ/mol)
      - arrhenius_log_k: float (computed Arrhenius feature value)
    """
    model = load_model()
    unc_model = load_uncertainty_model()

    with open(MODELS_DIR / "model_metadata.json") as f:
        meta = json.load(f)
    cal_factor = meta.get("uncertainty_calibration_factor", 1.0)

    input_df = pd.DataFrame([{
        "Catalyst": catalyst,
        "Catalyst_Dosage_wt%": dosage_wt_pct,
        "Temp_C": temp_c,
        "Time_min": time_min,
        "EG_PET_Ratio": eg_pet_ratio,
    }])

    X, _ = build_features(input_df, fit=False)
    prediction = float(model.predict(X)[0])

    # Compute uncertainty from RF tree spread
    X_arr = X.values if hasattr(X, "values") else X
    tree_preds = np.stack([tree.predict(X_arr) for tree in unc_model.estimators_])
    tree_std = float(tree_preds.std(axis=0)[0])
    half_width = tree_std * cal_factor * 1.28  # ~80% interval

    # Clamp to physically valid range [0, 100]
    lower = float(np.clip(prediction - half_width, 0, 100))
    upper = float(np.clip(prediction + half_width, 0, 100))

    # Retrieve physics features for display
    from feature_engineering import lookup_ea
    import json as _json
    with open(MODELS_DIR / "arrhenius_fallback_ea.json") as f:
        fallback_ea = _json.load(f)["fallback_ea_kj_mol"]
    ea_used = lookup_ea(catalyst) or fallback_ea
    arrhenius_log_k = -ea_used * 1000 / (8.314 * (temp_c + 273.15))

    return {
        "predicted_yield": round(prediction, 1),
        "lower": round(lower, 1),
        "upper": round(upper, 1),
        "activation_energy_used_kJ_mol": round(ea_used, 2),
        "arrhenius_log_k": round(arrhenius_log_k, 4),
    }


if __name__ == "__main__":
    result = predict_yield(
        catalyst="MnOx-500 nanorod",
        dosage_wt_pct=3.0,
        temp_c=180,
        time_min=180,
        eg_pet_ratio=20.0,
    )
    print(result)
