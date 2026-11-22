"""
feature_engineering.py
Physics-informed feature engineering: maps each catalyst to a literature
activation energy (Ea) where known, and computes an Arrhenius-derived
reactivity feature from the actual reaction temperature.

Rationale
---------
The Arrhenius equation states k = A * exp(-Ea / R*T): a catalyst's rate
constant grows exponentially as activation energy drops and/or temperature
rises. Two papers in our literature dataset (Zhang et al. 2024, Gabric et al.
2025) report measured activation energies for specific catalyst systems.
Rather than only feeding the model raw (Catalyst, Temp) as separate
unrelated columns, we compute a physically-motivated combined feature -
a relative Arrhenius reactivity term - that lets the model learn yield
trends consistent with chemical kinetics rather than purely from
catalyst-temperature correlations in 75 noisy rows.

For catalysts with no literature Ea (the majority - this is real, scarce
literature data), we impute the dataset's median known Ea so every row
still gets a defined feature value rather than dropping out of training.
"""

import re
import numpy as np
import pandas as pd

R_GAS_CONSTANT = 8.314  # J / (mol*K)

# Catalyst name (as it appears, or substring match) -> known activation
# energy in kJ/mol, extracted from the literature dataset's
# Activation_Energy_Comparison sheet. Using the CATALYZED step Ea where a
# paper reports multiple steps (catalyzed pathway is what's relevant once
# catalyst is present).
KNOWN_EA_KJ_MOL = {
    "zinc acetate": 158.0,          # Gabric et al. 2025, Ea2 (catalyzed, PET->intermediate)
    "[p4444]ace": 89.73,            # Zhang et al. 2024
    "cofe2o4": 188.0,
    "[ch][oac]": 131.31,
    "[bmim][oac]": 58.53,
    "[bmim]cl": 232.79,
    "cu(oac)2-[bmim][oac]": 56.4,
    "zn(oac)2-[bmim][oac]": 53.8,
    "[bmim-fe][(oac)3]/bentonite": 51.6,
    "(1,3-dmu)/zn(oac)2": 148.89,
    "t-bup2": 122.0,
}


def lookup_ea(catalyst_name: str):
    """Case-insensitive substring match against the known-Ea lookup table."""
    if pd.isna(catalyst_name):
        return None
    name = str(catalyst_name).lower()
    for key, ea in KNOWN_EA_KJ_MOL.items():
        if key in name:
            return ea
    return None


def add_arrhenius_features(df: pd.DataFrame, fallback_ea: float = None) -> pd.DataFrame:
    """
    Adds two physics-informed columns to the dataframe:
      - Ea_kJ_per_mol: literature activation energy for this catalyst
        (imputed with fallback_ea, typically the training-set median, if unknown)
      - Arrhenius_Reactivity: exp(-Ea / (R*T)) computed at the row's actual
        reaction temperature, scaled to a sensible numeric range for ML.

    fallback_ea must be supplied at inference time (the value learned/saved
    during training) to avoid leaking test-time information.
    """
    df = df.copy()
    df["Ea_kJ_per_mol"] = df["Catalyst"].apply(lookup_ea)

    if fallback_ea is None:
        fallback_ea = df["Ea_kJ_per_mol"].median()
        if pd.isna(fallback_ea):
            fallback_ea = 120.0  # sane generic mid-range Ea for glycolysis catalysts

    df["Ea_kJ_per_mol"] = df["Ea_kJ_per_mol"].fillna(fallback_ea)

    temp_k = df["Temp_C"].astype(float) + 273.15
    ea_j = df["Ea_kJ_per_mol"].astype(float) * 1000.0

    # log(k_relative) = -Ea/(R*T) -- using the log form keeps values in a
    # numerically stable range for tree-based models instead of the raw
    # (very small) exponential.
    df["Arrhenius_log_k"] = -ea_j / (R_GAS_CONSTANT * temp_k)

    return df, fallback_ea
