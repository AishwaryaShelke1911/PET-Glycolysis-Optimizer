"""
data_loader.py
Loads the literature-mined PET glycolysis dataset from Excel, cleans it,
and saves a ready-to-use CSV for modeling.

Run directly: python src/data_loader.py
"""

import re
import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "PET_Glycolysis_Dataset.xlsx"
PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "glycolysis_clean.csv"


def load_raw_sheet(sheet_name: str = "Glycolysis_Dataset") -> pd.DataFrame:
    """Load a single sheet from the raw Excel workbook."""
    df = pd.read_excel(RAW_PATH, sheet_name=sheet_name)
    return df


def parse_dosage(value, eg_ratio_text=None):
    """
    Extract numeric catalyst dosage in wt% where possible.

    Handles:
      - "15 wt%" -> 15.0
      - "0.5" (already a plain number) -> 0.5
      - "0.01g per 1g PET" -> 1.0  (computed from the two gram amounts)
      - "0.15 g" -> cross-references eg_ratio_text for a "...g PET" amount
        in the same row (e.g. "20mL EG/10g PET") to compute g_catalyst/g_PET*100

    Returns None if no reasonable numeric dosage could be determined.
    """
    if pd.isna(value):
        return None
    value = str(value)

    match = re.search(r"([\d.]+)\s*wt%", value)
    if match:
        return float(match.group(1))

    match = re.search(r"^([\d.]+)$", value.strip())
    if match:
        return float(match.group(1))

    # "0.01g per 1g PET" style - both amounts given in the same string
    match = re.search(r"([\d.]+)\s*g\s*per\s*([\d.]+)\s*g\s*PET", value, re.IGNORECASE)
    if match:
        cat_g, pet_g = float(match.group(1)), float(match.group(2))
        return (cat_g / pet_g) * 100 if pet_g != 0 else None

    # "0.15 g" / "0.10 g" style - need the PET amount from eg_ratio_text
    match = re.search(r"^([\d.]+)\s*g$", value.strip(), re.IGNORECASE)
    if match and eg_ratio_text and not pd.isna(eg_ratio_text):
        cat_g = float(match.group(1))
        pet_match = re.search(r"([\d.]+)\s*g\s*PET", str(eg_ratio_text), re.IGNORECASE)
        if pet_match:
            pet_g = float(pet_match.group(1))
            return (cat_g / pet_g) * 100 if pet_g != 0 else None

    return None


def parse_eg_ratio(value):
    """
    Attempt to extract a numeric EG:PET ratio (mass or molar basis) from free-text strings.
    Returns None if no ratio could be parsed - these rows may need manual cleanup.
    """
    if pd.isna(value):
        return None
    value = str(value)

    # Pattern like "5:1" or "1:6.9"
    match = re.search(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", value)
    if match:
        a, b = float(match.group(1)), float(match.group(2))
        return a / b if b != 0 else None

    # Pattern like "5g EG / 1g PET" or "50mL EG/5g colored PET" -> ratio = 5/1, 50/5
    match = re.search(r"([\d.]+)\s*(?:g|mL)\s*EG\s*/\s*([\d.]+)\s*g\s*(?:colored\s*)?PET", value, re.IGNORECASE)
    if match:
        a, b = float(match.group(1)), float(match.group(2))
        return a / b if b != 0 else None

    # Pattern like "500 wt% EG / 1g PET" -> 500% = 5x mass ratio
    match = re.search(r"([\d.]+)\s*wt%\s*EG", value, re.IGNORECASE)
    if match:
        return float(match.group(1)) / 100

    # Pattern like "5 (w/w EG:PET)" or "4 (EG/PET mass ratio)" -> leading number is the ratio itself
    match = re.search(r"^([\d.]+)\s*\(", value.strip())
    if match:
        return float(match.group(1))

    return None


def clean_numeric_column(series: pd.Series, max_sentinel: float = 9999) -> pd.Series:
    """
    Convert a column to numeric, coercing non-numeric / range values to NaN.
    Also treats placeholder sentinel values (e.g. 99999 used in the raw
    dataset to mean "very long / indefinite time") as missing, since they
    are not real measured values and would otherwise badly skew ranges
    used for grid-search optimization.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.where(numeric < max_sentinel)
    return numeric


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps and return a model-ready dataframe."""
    df = df.copy()

    df["Catalyst_Dosage_wt%"] = df.apply(
        lambda row: parse_dosage(row["Catalyst Dosage (wt% or other)"], row["EG:PET Ratio / EG dosage"]),
        axis=1,
    )
    df["EG_PET_Ratio"] = df["EG:PET Ratio / EG dosage"].apply(parse_eg_ratio)
    df["Temp_C"] = clean_numeric_column(df["Temp (°C)"])
    df["Time_min"] = clean_numeric_column(df["Time (min)"])
    df["PET_Conversion_pct"] = clean_numeric_column(df["PET Conversion (%)"])
    df["BHET_Yield_pct"] = clean_numeric_column(df["BHET Yield (%)"])

    # ── Normalise duplicate catalyst names ─────────────────────────────────
    # Several papers use slightly different names for the same catalyst system.
    # Merge them so the model treats them as one category.
    catalyst_aliases = {
        "Zinc acetate (ZnAc2)":                    "Zinc acetate",
        "Zinc acetate dihydrate":                   "Zinc acetate",
        "Zinc acetate dihydrate (uncatalyzed)":     None,   # control — drop
        "Blank (no catalyst)":                      None,   # control — drop
        "No DES (conventional heating)":            None,   # control — drop
        "Fe3O4-CP (coprecipitation)":               "Fe3O4-CP",
        "MgO/Y2O3 (reused cycle 2)":               "MgO/Y2O3",
        "MgO/Y2O3 (reused cycle 3)":               "MgO/Y2O3",
        "MgO/Y2O3 (1,2-propanediol/PG instead of EG)": "MgO/Y2O3",
        "MgO/Y2O3 (1,4-butanediol instead of EG)": "MgO/Y2O3",
        "MgO/Y2O3 (real waste: blue bottle)":      "MgO/Y2O3",
        "MgO/Y2O3 (real waste: green bottle)":     "MgO/Y2O3",
        "MgO/Y2O3 (real waste: packaging box)":    "MgO/Y2O3",
        "MgO/Y2O3 (real waste: transparent water bottle)": "MgO/Y2O3",
        "MnOx-500 nanorod (reused cycle 2)":       "MnOx-500 nanorod",
        "MnOx-500 nanorod (reused cycle 3)":       "MnOx-500 nanorod",
        "MnOx-500 nanorod (reused cycle 4)":       "MnOx-500 nanorod",
        "ChCl-Zn(OAc)2 DES + MW 1000W (EG dosage study)": "ChCl-Zn(OAc)2 DES + MW 1000W",
        "Fe3O4-CP (large scale)":                  "Fe3O4-CP",
        "Fe3O4-CP (reused 4x)":                    "Fe3O4-CP",
    }

    df["Catalyst"] = df["Catalyst"].map(
        lambda x: catalyst_aliases.get(x, x)
    )
    # Drop rows mapped to None (controls/blanks)
    df = df[df["Catalyst"].notna()]

    model_cols = [
        "Catalyst",
        "Catalyst_Dosage_wt%",
        "Temp_C",
        "Time_min",
        "EG_PET_Ratio",
        "PET_Conversion_pct",
        "BHET_Yield_pct",
        "Paper Citation",
    ]
    clean_df = df[model_cols].copy()

    # Drop rows with no target value at all (can't train or evaluate on these)
    clean_df = clean_df.dropna(subset=["BHET_Yield_pct"])

    return clean_df


def main():
    print(f"Loading raw data from {RAW_PATH} ...")
    raw_df = load_raw_sheet()
    print(f"Raw rows: {len(raw_df)}")

    clean_df = clean_dataset(raw_df)
    print(f"Rows with usable BHET yield after cleaning: {len(clean_df)}")

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved cleaned dataset to {PROCESSED_PATH}")
    print("\nPreview:")
    print(clean_df.head(10))
    print("\nMissing values per column:")
    print(clean_df.isna().sum())


if __name__ == "__main__":
    main()
