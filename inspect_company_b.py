
from src.spend_intelligence.config import load_config
from src.spend_intelligence.ingestion.loader import load_company_inputs

config = load_config("configs/company_b.yaml")

datasets = load_company_inputs(config)

for name, df in datasets.items():
    print("\n==========", name.upper(), "==========")
    print("Columns:")
    print(df.columns.tolist())
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())



from src.spend_intelligence.config import load_config
from src.spend_intelligence.ingestion.loader import load_company_inputs
from src.spend_intelligence.preprocessing.company_a import prepare_company_a

config = load_config("configs/company_a.yaml")
inputs = load_company_inputs(config)
processed = prepare_company_a(inputs, config)

tool = processed["invoice_tool"]

negative = tool[tool["spend_zar"] < 0]

print("Negative records:", len(negative))
print("\nAvailable columns:")
print(tool.columns.tolist())

print("\nNegative record details:")
print(
    negative.head(20).to_string(index=False)
)



from src.spend_intelligence.config import load_config
from src.spend_intelligence.ingestion.loader import load_company_inputs
from src.spend_intelligence.preprocessing.company_b import prepare_company_b

config = load_config("configs/company_b.yaml")
inputs = load_company_inputs(config)

processed = prepare_company_b(inputs, config)

df = processed["invoice_lines"]

columns = [
    "line_id",
    "invoice_number",
    "invoice_date",
    "currency",
    "local_amount",
    "currency_rate_numeric",
    "monthly_fx_rate",
    "fx_rate_used",
    "fx_source",
    "spend_usd"
]

print("\n========== MISSING FX RATES ==========")

missing_fx = df[df["fx_missing"]]

print("Count:", len(missing_fx))
print(missing_fx[columns].to_string(index=False))

print("\n========== MISSING USD SPEND ==========")

missing_spend = df[df["spend_missing"]]

print("Count:", len(missing_spend))
print(missing_spend[columns].to_string(index=False))

print("\n========== NEGATIVE AMOUNTS ==========")

negative = df[df["local_amount"] < 0]

print("Count:", len(negative))
print("\nDocument type distribution:")
print(negative["document_type"].value_counts(dropna=False))

print("\n========== CURRENCY DISTRIBUTION ==========")

print(df["currency"].value_counts(dropna=False))



from src.spend_intelligence.config import load_config
from src.spend_intelligence.ingestion.loader import load_company_inputs
from src.spend_intelligence.preprocessing.company_b import prepare_company_b

config = load_config("configs/company_b.yaml")
inputs = load_company_inputs(config)

processed = prepare_company_b(inputs, config)

df = processed["invoice_lines"]

# -----------------------------------------
# 1. Inspect all eight problematic records
# -----------------------------------------

exception_ids = [
    "B005026",
    "B005027",
    "B005028",
    "B005029",
    "B005030",
    "B025101",
    "B025867",
    "B026435"
]

exceptions = df[df["line_id"].isin(exception_ids)].copy()

labels = inputs["labels"].copy()

exceptions = exceptions.merge(
    labels,
    on="line_id",
    how="left",
    indicator=True,
    suffixes=("", "_label")
)

print("\n========== EXCEPTION RECORDS + SEED LABELS ==========")

columns = [
    "line_id",
    "invoice_number",
    "invoice_date",
    "currency",
    "local_amount",
    "fx_rate_used",
    "spend_usd",
    "document_type",
    "include",
    "exclusion_reason",
    "category_l1",
    "category_l2",
    "category_l3",
    "_merge"
]

print(
    exceptions[
        [c for c in columns if c in exceptions.columns]
    ].to_string(index=False)
)

# -----------------------------------------
# 2. Inspect CAD monthly FX coverage
# -----------------------------------------

fx = inputs["fx_rates"].copy()

fx["month"] = fx["month"].astype("string").str.strip()

cad_rates = fx[
    fx["currency"].astype("string").str.upper() == "CAD"
].sort_values("month")

print("\n========== CAD FX COVERAGE ==========")
print(cad_rates.to_string(index=False))

# -----------------------------------------
# 3. Inspect RCN currency in source data
# -----------------------------------------

print("\n========== RCN SOURCE RECORDS ==========")

rcn = df[df["currency"].astype("string").str.upper() == "RCN"]

print(
    rcn[
        [
            "line_id",
            "invoice_number",
            "invoice_date",
            "currency",
            "local_amount",
            "currency_rate_numeric",
            "monthly_fx_rate",
            "document_type",
            "vendor_id",
            "source_system",
            "region"
        ]
    ].to_string(index=False)
)

print("\nRCN records in seed labels:")
print(
    labels[
        labels["line_id"].isin(rcn["line_id"])
    ].to_string(index=False)
)



import pandas as pd

from src.spend_intelligence.config import load_config
from src.spend_intelligence.ingestion.loader import load_company_inputs
from src.spend_intelligence.preprocessing.company_b import (
    prepare_company_b,
    apply_company_b_inclusion,
)

config = load_config("configs/company_b.yaml")
inputs = load_company_inputs(config)

prepared = prepare_company_b(inputs, config)
predictions = apply_company_b_inclusion(
    prepared["invoice_lines"]
)

# Compare predictions with the known seed labels
labels = inputs["labels"].copy()
labels = labels.rename(columns={"include": "actual_include"})

comparison = labels.merge(
    predictions[["line_id", "include"]],
    on="line_id",
    how="left",
    validate="one_to_one",
)

comparison["actual_include"] = pd.to_numeric(
    comparison["actual_include"], errors="coerce"
)
comparison["predicted_include"] = pd.to_numeric(
    comparison["include"], errors="coerce"
)

valid = comparison.dropna(
    subset=["actual_include", "predicted_include"]
)

print("\n========== COMPANY B VALIDATION ==========")
print("Seed labels:", len(labels))
print("Matched records:", len(valid))

print("\nConfusion matrix:")
print(
    pd.crosstab(
        valid["actual_include"],
        valid["predicted_include"],
        rownames=["Actual"],
        colnames=["Predicted"],
    )
)

accuracy = (
    valid["actual_include"] == valid["predicted_include"]
).mean()

print(f"\nAccuracy: {accuracy:.4f}")

errors = valid[
    valid["actual_include"] != valid["predicted_include"]
]

print("\nIncorrect predictions:", len(errors))

if not errors.empty:
    print(
        errors[
            ["line_id", "actual_include", "predicted_include"]
        ].head(20).to_string(index=False)
    )

print("\nValidation completed.")


# Inspect all incorrectly included records
error_details = errors.merge(
    inputs["invoice_lines"],
    on="line_id",
    how="left"
)

print("\n========== 59 FALSE INCLUSIONS ==========")

columns = [
    "line_id",
    "line_description",
    "material_number",
    "supplier_type",
    "gl_account",
    "gl_account_description",
    "cost_centre",
    "document_type",
    "po_type",
    "exclusion_reason"
]

print(
    error_details[columns].to_string(index=False)
)

print("\nGL account description distribution:")
print(
    error_details["gl_account_description"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nSupplier type distribution:")
print(
    error_details["supplier_type"]
    .value_counts(dropna=False)
    .to_string()
)