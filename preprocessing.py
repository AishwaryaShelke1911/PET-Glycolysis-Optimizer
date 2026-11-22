"""
preprocessing.py
Feature engineering pipeline: encodes categorical catalyst column, adds
physics-informed Arrhenius features, imputes missing numeric values, and
scales features for ML.
"""

import json
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from feature_engineering import add_arrhenius_features

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

BASE_NUMERIC_FEATURES = ["Catalyst_Dosage_wt%", "Temp_C", "Time_min", "EG_PET_Ratio"]
PHYSICS_FEATURES = ["Ea_kJ_per_mol", "Arrhenius_log_k"]
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + PHYSICS_FEATURES
CATEGORICAL_FEATURES = ["Catalyst"]
TARGET = "BHET_Yield_pct"


def build_features(df: pd.DataFrame, fit: bool = True):
    """
    Build the feature matrix X and target vector y from the cleaned dataframe.

    If fit=True, fits new encoders/imputers/scalers and saves them to models/.
    If fit=False, loads existing encoders/imputers/scalers from models/ (for inference).
    """
    df = df.copy()

    # Arrhenius features need raw Temp_C (pre-scaling) and Catalyst (pre-encoding)
    if fit:
        df, fallback_ea = add_arrhenius_features(df, fallback_ea=None)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODELS_DIR / "arrhenius_fallback_ea.json", "w") as f:
            json.dump({"fallback_ea_kj_mol": float(fallback_ea)}, f)
    else:
        with open(MODELS_DIR / "arrhenius_fallback_ea.json") as f:
            fallback_ea = json.load(f)["fallback_ea_kj_mol"]
        df, _ = add_arrhenius_features(df, fallback_ea=fallback_ea)

    if fit:
        cat_encoder = LabelEncoder()
        df["Catalyst_encoded"] = cat_encoder.fit_transform(df["Catalyst"].astype(str))

        imputer = SimpleImputer(strategy="median")
        df[NUMERIC_FEATURES] = imputer.fit_transform(df[NUMERIC_FEATURES])

        scaler = StandardScaler()
        df[NUMERIC_FEATURES] = scaler.fit_transform(df[NUMERIC_FEATURES])

        joblib.dump(cat_encoder, MODELS_DIR / "catalyst_encoder.pkl")
        joblib.dump(imputer, MODELS_DIR / "imputer.pkl")
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    else:
        cat_encoder = joblib.load(MODELS_DIR / "catalyst_encoder.pkl")
        imputer = joblib.load(MODELS_DIR / "imputer.pkl")
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")

        # Handle unseen catalysts gracefully at inference time
        known_classes = set(cat_encoder.classes_)
        df["Catalyst"] = df["Catalyst"].astype(str).apply(
            lambda x: x if x in known_classes else cat_encoder.classes_[0]
        )
        df["Catalyst_encoded"] = cat_encoder.transform(df["Catalyst"])
        df[NUMERIC_FEATURES] = imputer.transform(df[NUMERIC_FEATURES])
        df[NUMERIC_FEATURES] = scaler.transform(df[NUMERIC_FEATURES])

    feature_cols = NUMERIC_FEATURES + ["Catalyst_encoded"]
    X = df[feature_cols]
    y = df[TARGET] if TARGET in df.columns else None

    return X, y


def get_catalyst_list():
    """Return the list of known catalyst names (for populating the app dropdown)."""
    encoder = joblib.load(MODELS_DIR / "catalyst_encoder.pkl")
    return sorted(encoder.classes_.tolist())
