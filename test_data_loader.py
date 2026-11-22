"""
test_data_loader.py
Basic unit tests for the data cleaning/parsing functions.
Run: pytest tests/
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from data_loader import parse_dosage, parse_eg_ratio, clean_numeric_column


def test_parse_dosage_with_wt_percent():
    assert parse_dosage("15 wt%") == 15.0
    assert parse_dosage("1 wt%") == 1.0


def test_parse_dosage_plain_number():
    assert parse_dosage("0.5") == 0.5


def test_parse_dosage_none_for_unparseable():
    assert parse_dosage("-") is None
    assert parse_dosage(None) is None


def test_parse_dosage_grams_per_grams():
    assert parse_dosage("0.01g per 1g PET") == 1.0


def test_parse_dosage_cross_reference_eg_column():
    # 0.15 g catalyst, and EG ratio column mentions "10g PET" -> 1.5 wt%
    assert parse_dosage("0.15 g", "20mL EG/10g PET (1:6.9)") == 1.5


def test_parse_eg_ratio_colon_format():
    assert parse_eg_ratio("1:6.9 PET:EG molar") == 1 / 6.9


def test_parse_eg_ratio_g_format():
    assert parse_eg_ratio("5g EG / 1g PET") == 5.0


def test_clean_numeric_column_handles_text():
    series = pd.Series(["100", "low", "64.2", None])
    cleaned = clean_numeric_column(series)
    assert cleaned[0] == 100.0
    assert pd.isna(cleaned[1])
    assert cleaned[2] == 64.2
    assert pd.isna(cleaned[3])
