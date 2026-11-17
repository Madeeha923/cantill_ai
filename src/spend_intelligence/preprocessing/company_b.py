
import pandas as pd
import numpy as np

from .cleaning import (
    clean_text,
    parse_numeric,
    parse_dates
)


def prepare_company_b(inputs: dict, config: dict) -> dict:
    """
    Preprocess Company B invoice lines and monthly FX rates.

    - Clean text and numeric fields.
    - Parse invoice dates.
    - Use transaction-level currency_rate when valid.
    - Fall back to monthly FX rates when currency_rate is missing.
    - Use 1.0 for USD when no rate is supplied.
    - Convert local spend into USD.
    - Preserve all original records.
    """

    company_config = config["company_b"]

    invoices = inputs["invoice_lines"].copy()
    fx = inputs["fx_rates"].copy()

    # -----------------------------------------
    # 1. Clean invoice text fields
    # -----------------------------------------

    text_columns = [
        "line_id",
        "currency",
        "invoice_number",
        "line_description",
        "material_number",
        "vendor_id",
        "document_type",
        "gl_account",
        "gl_account_description",
        "source_system",
        "region"
    ]

    for column in text_columns:
        if column in invoices.columns:
            invoices[column] = clean_text(invoices[column])

    # -----------------------------------------
    # 2. Parse invoice dates
    # -----------------------------------------

    invoices = parse_dates(
        invoices,
        [
            "posting_date",
            "invoice_date",
            "due_date",
            "paid_date"
        ]
    )

    # Invoice month is used for monthly FX lookup
    invoices["invoice_month"] = (
        invoices["invoice_date"].dt.strftime("%Y-%m")
    )

    # -----------------------------------------
    # 3. Convert invoice amount and rate
    # -----------------------------------------

    amount_column = company_config["amount_column"]
    currency_column = company_config["currency_column"]
    rate_column = company_config["rate_column"]

    invoices["local_amount"], invoices["amount_parse_error"] = (
        parse_numeric(invoices[amount_column])
    )

    invoices["currency_rate_numeric"], rate_parse_error = (
        parse_numeric(invoices[rate_column])
    )

    invoices["rate_parse_error"] = rate_parse_error

    # -----------------------------------------
    # 4. Prepare monthly FX rates
    # -----------------------------------------

    fx_currency_column = company_config["fx_currency_column"]
    fx_month_column = company_config["fx_month_column"]
    fx_rate_column = company_config["fx_rate_column"]

    fx[fx_currency_column] = clean_text(
        fx[fx_currency_column]
    ).str.upper()

    fx[fx_month_column] = (
        fx[fx_month_column]
        .astype("string")
        .str.strip()
        .str[:7]
    )

    fx["monthly_fx_rate"], fx["fx_parse_error"] = (
        parse_numeric(fx[fx_rate_column])
    )

    fx = fx.rename(
        columns={
            fx_currency_column: "fx_currency",
            fx_month_column: "fx_month"
        }
    )

    # Ignore invalid monthly rates
    fx.loc[
        fx["monthly_fx_rate"] <= 0,
        "monthly_fx_rate"
    ] = np.nan

    # Each currency-month combination should have one rate
    duplicate_fx = fx.duplicated(
        subset=["fx_currency", "fx_month"],
        keep=False
    )

    if duplicate_fx.any():
        raise ValueError(
            "Duplicate currency-month combinations found "
            "in the FX rates file. Please inspect the FX dataset."
        )

    fx_lookup = fx[
        ["fx_currency", "fx_month", "monthly_fx_rate"]
    ].copy()

    # -----------------------------------------
    # 5. Join monthly FX rates
    # -----------------------------------------

    invoices["_fx_currency"] = (
        invoices[currency_column]
        .astype("string")
        .str.upper()
    )

    invoices["_fx_month"] = invoices["invoice_month"]

    invoices = invoices.merge(
        fx_lookup,
        how="left",
        left_on=["_fx_currency", "_fx_month"],
        right_on=["fx_currency", "fx_month"],
        validate="many_to_one"
    )

    # -----------------------------------------
    # 6. Select the rate to use
    # -----------------------------------------

    row_rate = invoices["currency_rate_numeric"]

    valid_row_rate = (
        row_rate.notna() &
        (row_rate > 0)
    )

    valid_monthly_rate = (
        invoices["monthly_fx_rate"].notna() &
        (invoices["monthly_fx_rate"] > 0)
    )

    is_usd = invoices["_fx_currency"].eq("USD")

    invoices["fx_rate_used"] = np.nan
    invoices["fx_source"] = "MISSING"

    # USD values are already in the reporting currency. A supplied row rate
    # must not scale them (some USD rows carry a non-1 currency_rate value).
    invoices.loc[is_usd, "fx_rate_used"] = 1.0
    invoices.loc[is_usd, "fx_source"] = "USD_PASSTHROUGH"

    # Priority 1: transaction-level rate
    invoices.loc[
        valid_row_rate & ~is_usd,
        "fx_rate_used"
    ] = row_rate[valid_row_rate & ~is_usd]

    invoices.loc[
        valid_row_rate & ~is_usd,
        "fx_source"
    ] = "TRANSACTION_RATE"

    # Priority 2: monthly FX fallback
    use_monthly_fx = (
        ~is_usd &
        ~valid_row_rate &
        valid_monthly_rate
    )

    invoices.loc[
        use_monthly_fx,
        "fx_rate_used"
    ] = invoices.loc[
        use_monthly_fx,
        "monthly_fx_rate"
    ]

    invoices.loc[
        use_monthly_fx,
        "fx_source"
    ] = "MONTHLY_FX"

    # -----------------------------------------
    # 7. Convert local amount to USD
    # -----------------------------------------

    invoices["spend_usd"] = (
        invoices["local_amount"] /
        invoices["fx_rate_used"]
    )

    # Missing FX flag for non-USD currencies
    invoices["amount_missing"] = invoices["local_amount"].isna()
    invoices["fx_missing"] = ~is_usd & invoices["fx_rate_used"].isna()

    # Missing spend diagnostics
    invoices["spend_missing"] = invoices["amount_missing"] | invoices["fx_missing"]

    # -----------------------------------------
    # 8. Return processed datasets
    # -----------------------------------------

    invoices = invoices.drop(
        columns=[
            "_fx_currency",
            "_fx_month",
            "fx_currency",
            "fx_month"
        ],
        errors="ignore"
    )

    return {
        "invoice_lines": invoices,
        "fx_rates": fx
    }



import pandas as pd


def apply_company_b_inclusion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply conservative procurement inclusion rules for Company B.

    Exclusion evidence:
    1. Intercompany GL account descriptions.
    2. Explicit sales tax descriptions.

    All other records remain included by default.
    """

    result = df.copy()

    gl_description = (
        result["gl_account_description"]
        .astype("string")
        .str.strip()
    )

    line_description = (
        result["line_description"]
        .astype("string")
        .str.strip()
    )

    # Rule 1: Intercompany transactions
    intercompany_mask = gl_description.str.match(
        r"^I\s*/?\s*C",
        case=False,
        na=False
    )

    tax_mask = line_description.str.contains(
    r"\btax(?:es)?\b",
    case=False,
    na=False,
    regex=True
)


    # Initialize all records as included
    result["include"] = 1
    result["exclusion_reason"] = ""

    # Apply exclusions
    exclusion_mask = intercompany_mask | tax_mask

    result.loc[exclusion_mask, "include"] = 0
    result.loc[exclusion_mask, "exclusion_reason"] = (
        "NON_PROCUREMENT"
    )

    # Keep the reason blank for included records
    result.loc[result["include"] == 1, "exclusion_reason"] = ""

    return result
