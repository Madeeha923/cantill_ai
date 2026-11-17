
import pandas as pd
import numpy as np


def combine_fields(row, columns):
    """Join configured row fields while skipping null and blank values."""
    values = []
    for column in columns:
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            values.append(str(value).strip())
    return " ".join(values)


def canonical_item(description=None, code=None):
    """Prefer a stable item code; use description when no code is available."""
    for value in (code, description):
        if pd.notna(value) and str(value).strip().lower() not in {"", "nan", "none", "null"}:
            return str(value).strip()
    return ""

def clean_text(series: pd.Series) -> pd.Series:
    """
    Standardize text values:
    - Remove leading/trailing spaces
    - Convert empty strings to missing values
    - Normalize common missing-value placeholders
    """

    result = series.astype("string").str.strip()

    missing_values = [
        "",
        "nan",
        "none",
        "null",
        "n/a",
        "na",
        "unknown"
    ]

    result = result.mask(
        result.str.lower().isin(missing_values),
        pd.NA
    )

    return result

def parse_numeric(series: pd.Series) -> tuple:
    """
    Convert text-based numeric values into float.

    Handles:
    - 5,266.63
    - 5266.63
    - (5266.63) as -5266.63
    - Currency symbols and surrounding spaces

    Returns:
        numeric_values
        invalid_value_mask
    """

    raw = series.astype("string").str.strip()

    # Remove thousands separators
    normalized = raw.str.replace(",", "", regex=False)

    # Remove common currency symbols and other characters
    normalized = normalized.str.replace(
        r"[^0-9.()+-]",
        "",
        regex=True
    )

    # Detect accounting-style negative values
    accounting_negative = normalized.str.match(
        r"^\(.*\)$",
        na=False
    )

    # Remove parentheses before conversion
    normalized = normalized.str.replace(
        r"^\((.*)\)$",
        r"\1",
        regex=True
    )

    numeric_values = pd.to_numeric(
        normalized,
        errors="coerce"
    )

    numeric_values = numeric_values.where(
        ~accounting_negative,
        -numeric_values.abs()
    )

    # Flag non-empty values that failed conversion
    invalid_mask = raw.notna() & numeric_values.isna()

    return numeric_values.astype("float64"), invalid_mask

def parse_dates(
    df: pd.DataFrame,
    columns: list
) -> pd.DataFrame:
    """
    Parse mixed-format date columns.
    Invalid or missing dates become NaT.
    Original rows are preserved.
    """

    result = df.copy()

    for column in columns:

        if column in result.columns:

            result[column] = pd.to_datetime(
                result[column],
                format="mixed",
                errors="coerce"
            )

    return result
