
from src.spend_intelligence.config import load_config
from src.spend_intelligence.ingestion.loader import load_company_inputs

from src.spend_intelligence.preprocessing.company_a import (
    prepare_company_a
)


# Load configuration
config = load_config("configs/company_a.yaml")

# Load raw datasets
inputs = load_company_inputs(config)

# Preprocess Company A
processed = prepare_company_a(inputs, config)

erp = processed["erp"]
tool = processed["invoice_tool"]


print("\n========== ERP RESULTS ==========")

print("Rows:", len(erp))
print("Missing spend:", erp["spend_zar"].isna().sum())
print("Negative spend:", (erp["spend_zar"] < 0).sum())
print("Amount parsing errors:", erp["amount_parse_error"].sum())

print("\nERP sample:")
print(erp[["spend_zar", "amount_parse_error"]].head())


print("\n========== INVOICE TOOL RESULTS ==========")

print("Rows:", len(tool))
print("Missing net spend:", tool["spend_zar"].isna().sum())
print("Negative spend:", (tool["spend_zar"] < 0).sum())
print("VAT defaulted:", tool["vat_rate_defaulted"].sum())
print("Amount parsing errors:", tool["amount_parse_error"].sum())

print("\nInvoice Tool sample:")
print(
    tool[
        [
            "gross_amount_numeric",
            "vat_rate_used",
            "spend_zar"
        ]
    ].head()
)

from src.spend_intelligence.config import load_config
from src.spend_intelligence.ingestion.loader import load_company_inputs

config = load_config("configs/company_a.yaml")
data = load_company_inputs(config)

for name in ["erp", "invoice_tool"]:
    df = data[name]

    print(f"\n{'=' * 60}")
    print(f"{name.upper()}")
    print(f"{'=' * 60}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nSample records:")
    print(df.head(5).to_string(index=False))

import re
import pandas as pd

from src.spend_intelligence.config import load_config
from src.spend_intelligence.ingestion.loader import load_company_inputs
from src.spend_intelligence.preprocessing.company_a import prepare_company_a


config = load_config("configs/company_a.yaml")
inputs = load_company_inputs(config)
data = prepare_company_a(inputs, config)

erp = data["erp"].copy()
tool = data["invoice_tool"].copy()


def normalize_po(value):
    if pd.isna(value):
        return None
    match = re.search(r"\d{8,}", str(value))
    return match.group(0) if match else None


def normalize_item(value):
    if pd.isna(value):
        return None
    try:
        return str(int(float(value)))
    except (ValueError, TypeError):
        return None


def normalize_supplier(value):
    if pd.isna(value):
        return None
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


# Build ERP matching keys
erp["po_key"] = erp["po_number"].apply(normalize_po)
erp["item_key_match"] = erp["po_item"].apply(normalize_item)
erp["supplier_match"] = erp["supplier_name"].apply(normalize_supplier)

# Extract PO number and item from Invoice Tool references
tool["po_key"] = tool["po_reference"].apply(normalize_po)

tool["item_key_match"] = tool["po_reference"].astype("string").str.extract(
    r"/\s*(\d+(?:\.0)?)"
)[0].apply(normalize_item)

tool["supplier_match"] = tool["supplier_name_as_captured"].apply(
    normalize_supplier
)

# Compare PO-only and PO+item matches
erp_po = set(erp["po_key"].dropna())
erp_po_item = set(
    zip(
        erp.loc[erp["po_key"].notna(), "po_key"],
        erp.loc[erp["po_key"].notna(), "item_key_match"]
    )
)

tool["po_match"] = tool["po_key"].isin(erp_po)

tool["po_item_match"] = [
    (po, item) in erp_po_item if po and item else False
    for po, item in zip(tool["po_key"], tool["item_key_match"])
]

print("\n========== MATCH SUMMARY ==========")
print("Invoice Tool rows:", len(tool))
print("Rows with PO reference:", tool["po_key"].notna().sum())
print("PO number matches:", tool["po_match"].sum())
print("PO + item matches:", tool["po_item_match"].sum())

# Check whether PO+item identifies multiple ERP receipt rows
erp_key_counts = (
    erp.dropna(subset=["po_key", "item_key_match"])
    .groupby(["po_key", "item_key_match"])
    .size()
)

print("\nERP PO+item keys:", len(erp_key_counts))
print("Keys with multiple ERP rows:", (erp_key_counts > 1).sum())

print("\n========== MATCHED EXAMPLES ==========")
print(
    tool.loc[
        tool["po_item_match"],
        ["tool_record_id", "po_reference", "supplier_name_as_captured",
         "spend_zar", "po_key", "item_key_match"]
    ].head(10).to_string(index=False)
)

print("\n========== UNMATCHED EXAMPLES ==========")
print(
    tool.loc[
        ~tool["po_item_match"],
        ["tool_record_id", "po_reference", "supplier_name_as_captured",
         "spend_zar"]
    ].head(10).to_string(index=False)
)