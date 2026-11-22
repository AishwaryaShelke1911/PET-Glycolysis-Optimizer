"""
train_model.py
Trains and compares multiple regression models to predict BHET yield from
reaction conditions (including physics-informed Arrhenius features), using
proper k-fold cross-validation rather than a single train/test split given
the small literature-mined dataset size. Also calibrates a Random-Forest-
based prediction interval (uncertainty band) for use at inference time.

Run: python src/train_model.py
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from preprocessing import build_features, NUMERIC_FEATURES

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "glycolysis_clean.csv"
MODELS_DIR = BASE_DIR / "models"

N_FOLDS = 5
RANDOM_STATE = 42


def load_data():
    return pd.read_csv(PROCESSED_PATH)


def get_models():
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=5,
                                                min_samples_leaf=3, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=150, max_depth=2,
                                                         learning_rate=0.05, random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            min_child_weight=3, random_state=RANDOM_STATE
        )
    return models


def run_cross_validation(X, y, n_folds=N_FOLDS):
    """
    Run k-fold cross-validation for every candidate model and report
    mean +/- std R2 / RMSE / MAE across folds, rather than a single
    train/test split number (which is statistically unreliable on a
    dataset this size - a single split's score can swing wildly
    depending on which rows land in the test fold).
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "r2": "r2",
        "neg_rmse": "neg_root_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
    }

    results = {}
    for name, model in get_models().items():
        cv_result = cross_validate(model, X, y, cv=kf, scoring=scoring)
        results[name] = {
            "r2_mean": float(np.mean(cv_result["test_r2"])),
            "r2_std": float(np.std(cv_result["test_r2"])),
            "rmse_mean": float(-np.mean(cv_result["test_neg_rmse"])),
            "rmse_std": float(np.std(cv_result["test_neg_rmse"])),
            "mae_mean": float(-np.mean(cv_result["test_neg_mae"])),
            "mae_std": float(np.std(cv_result["test_neg_mae"])),
        }
        print(f"{name:20s}  R2={results[name]['r2_mean']:+.3f}±{results[name]['r2_std']:.3f}"
              f"   RMSE={results[name]['rmse_mean']:.2f}±{results[name]['rmse_std']:.2f}"
              f"   MAE={results[name]['mae_mean']:.2f}±{results[name]['mae_std']:.2f}")
    return results


def calibrate_uncertainty(rf_model, X, y, test_size=0.2):
    """
    Estimate the Random Forest's typical prediction-interval width by
    looking at the spread of individual tree predictions on a held-out
    split, and comparing that spread to actual residual error. This gives
    a rough calibration factor so the app can report an honest +/- range
    around each prediction instead of a bare point estimate.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    rf_model.fit(X_train, y_train)

    X_test_arr = X_test.values if hasattr(X_test, "values") else X_test
    tree_preds = np.stack([tree.predict(X_test_arr) for tree in rf_model.estimators_])
    tree_std = tree_preds.std(axis=0)
    mean_pred = tree_preds.mean(axis=0)
    residual = np.abs(y_test.values - mean_pred)

    # Calibration factor: how many tree-std units cover the typical
    # residual (so the app can scale the raw RF tree spread into a more
    # realistic uncertainty band).
    avg_tree_std = float(np.mean(tree_std)) if np.mean(tree_std) > 1e-6 else 1.0
    avg_residual = float(np.mean(residual))
    calibration_factor = avg_residual / avg_tree_std

    # Refit on full data for final deployed uncertainty model
    rf_model.fit(X, y)

    return rf_model, calibration_factor


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows for training.")

    if len(df) < 30:
        print("NOTE: This is a small, literature-mined dataset (not a lab-scale "
              "experimental dataset). K-fold cross-validation is used instead of "
              "a single train/test split because a single split's R2 can vary "
              "drastically depending on which rows happen to land in the test set.")

    X, y = build_features(df, fit=True)

    print(f"\n5-fold cross-validation results (mean +/- std across folds):\n")
    cv_results = run_cross_validation(X, y)

    best_name = max(cv_results, key=lambda k: cv_results[k]["r2_mean"])
    print(f"\nBest model by mean CV R2: {best_name}")

    # Fit best model on the FULL dataset for deployment (more data = better
    # final model; CV above was only for honest performance estimation)
    models = get_models()
    best_model = models[best_name]
    best_model.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")

    # Separately train + calibrate a Random Forest specifically for
    # uncertainty quantification (used by predict.py to report a +/- range
    # even when the deployed best_model is XGBoost, which doesn't expose
    # per-tree variance the same way).
    rf_for_uncertainty = RandomForestRegressor(
        n_estimators=300, max_depth=5, min_samples_leaf=3, random_state=RANDOM_STATE
    )
    rf_for_uncertainty, calibration_factor = calibrate_uncertainty(rf_for_uncertainty, X, y)
    joblib.dump(rf_for_uncertainty, MODELS_DIR / "uncertainty_model.pkl")

    metadata = {
        "best_model": best_name,
        "cv_results": cv_results,
        "n_folds": N_FOLDS,
        "features": NUMERIC_FEATURES + ["Catalyst_encoded"],
        "n_rows_total": len(X),
        "uncertainty_calibration_factor": calibration_factor,
        "notes": (
            "Metrics are 5-fold cross-validated mean+/-std, not a single "
            "train/test split, due to small literature-mined sample size. "
            "Includes physics-informed Arrhenius activation-energy features "
            "derived from literature kinetic data where available."
        ),
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved best model ({best_name}) to {MODELS_DIR / 'best_model.pkl'}")
    print(f"Saved uncertainty model to {MODELS_DIR / 'uncertainty_model.pkl'}")
    print(f"Saved metadata to {MODELS_DIR / 'model_metadata.json'}")

    if hasattr(best_model, "feature_importances_"):
        importances = dict(zip(X.columns, best_model.feature_importances_))
        print("\nFeature importances:")
        for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
            print(f"  {feat:25s} {imp:.3f}")


if __name__ == "__main__":
    main()
