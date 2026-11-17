
import pandas as pd

from .cleaning import (
    clean_text,
    parse_numeric,
    parse_dates
)


def prepare_company_a(inputs: dict, config: dict) -> dict:
    """
    Preprocess Company A ERP and Invoice Tool datasets.

    ERP:
        Keep the existing net amount in ZAR.

    Invoice Tool:
        Convert gross amount to net amount by removing VAT.

    No records are excluded or deleted here.
    """

    erp = inputs["erp"].copy()
    tool = inputs["invoice_tool"].copy()

    company_config = config["company_a"]

    # -----------------------------------------
    # 1. ERP preprocessing
    # -----------------------------------------

    erp_amount_column = company_config["erp_amount_column"]

    erp["spend_zar"], erp["amount_parse_error"] = (
        parse_numeric(erp[erp_amount_column])
    )

    # Parse ERP date
    erp = parse_dates(
        erp,
        ["posting_date"]
    )

    # Clean available text fields
    erp_text_columns = [
        "supplier_name",
        "vendor_id",
        "material_number",
        "material_short_text",
        "service_short_text"
    ]

    for column in erp_text_columns:
        if column in erp.columns:
            erp[column] = clean_text(erp[column])

    # -----------------------------------------
    # 2. Invoice Tool preprocessing
    # -----------------------------------------

    gross_column = company_config["invoice_tool_amount_column"]
    vat_column = company_config["invoice_tool_vat_column"]

    # Convert gross amount to numeric
    tool["gross_amount_numeric"], tool["amount_parse_error"] = (
        parse_numeric(tool[gross_column])
    )

    # Convert VAT rate to numeric
    vat_numeric, vat_parse_error = parse_numeric(
        tool[vat_column]
    )

    # Support both 15 and 0.15 representations
    vat_rate = vat_numeric.where(
        vat_numeric.isna() | (vat_numeric <= 1),
        vat_numeric / 100
    )

    # Apply default VAT rate when missing or invalid
    default_vat = float(
        company_config["default_vat_rate"]
    )

    tool["vat_rate_used"] = vat_rate.fillna(default_vat)

    tool["vat_rate_defaulted"] = (
        vat_rate.isna()
    )

    # Remove VAT from gross amount
    tool["spend_zar"] = (
        tool["gross_amount_numeric"] /
        (1 + tool["vat_rate_used"])
    )

    # Keep parsing diagnostics
    tool["vat_parse_error"] = vat_parse_error

    # Parse invoice date
    tool = parse_dates(
        tool,
        ["invoice_date"]
    )

    # Clean available text fields
    tool_text_columns = [
        "supplier_name_as_captured",
        "line_text",
        "invoice_number",
        "po_reference"
    ]

    for column in tool_text_columns:
        if column in tool.columns:
            tool[column] = clean_text(tool[column])

    # -----------------------------------------
    # 3. Return processed datasets
    # -----------------------------------------

    return {
        "erp": erp,
        "invoice_tool": tool
    }
