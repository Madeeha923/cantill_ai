
from src.spend_intelligence.config import load_config
from src.spend_intelligence.ingestion.loader import load_company_inputs
from src.spend_intelligence.preprocessing.company_b import (
    prepare_company_b
)

config = load_config("configs/company_b.yaml")

inputs = load_company_inputs(config)

processed = prepare_company_b(inputs, config)

df = processed["invoice_lines"]

print("\n========== COMPANY B RESULTS ==========")

print("Total rows:", len(df))
print("Missing local amounts:", df["local_amount"].isna().sum())
print("Amount parsing errors:", df["amount_parse_error"].sum())

print("\nFX source distribution:")
print(df["fx_source"].value_counts(dropna=False))

print("\nMissing FX rates:", df["fx_missing"].sum())
print("Missing USD spend:", df["spend_missing"].sum())

print("\nNegative local amounts:", (df["local_amount"] < 0).sum())
print("Negative USD spend:", (df["spend_usd"] < 0).sum())

print("\nSample converted records:")
print(
    df[
        [
            "line_id",
            "currency",
            "local_amount",
            "currency_rate_numeric",
            "monthly_fx_rate",
            "fx_rate_used",
            "fx_source",
            "spend_usd"
        ]
    ].head(10).to_string(index=False)
)



from src.spend_intelligence.config import load_config
from src.spend_intelligence.ingestion.loader import load_company_inputs
from src.spend_intelligence.preprocessing.company_b import (
    prepare_company_b,
    apply_company_b_inclusion
)

config = load_config("configs/company_b.yaml")
inputs = load_company_inputs(config)

# Run existing preprocessing
prepared = prepare_company_b(inputs, config)
invoice_lines = prepared["invoice_lines"]

# Apply inclusion rules
result = apply_company_b_inclusion(invoice_lines)

print("\n========== COMPANY B INCLUSION RESULTS ==========")

print("Total records:", len(result))

print("\nInclusion distribution:")
print(result["include"].value_counts().to_string())

print("\nExclusion reasons:")
print(
    result["exclusion_reason"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nExcluded by rule:")
print(
    "Intercompany:",
    result["gl_account_description"]
    .astype("string")
    .str.match(r"^I\s*/?\s*C", case=False, na=False)
    .sum()
)

print(
    "Sales tax:",
    result["line_description"]
    .astype("string")
    .str.contains(
        r"\bsales\s+tax\b",
        case=False,
        na=False
    ).sum()
)

print("\nExcluded examples:")
print(
    result.loc[
        result["include"] == 0,
        [
            "line_id",
            "gl_account",
            "gl_account_description",
            "line_description",
            "include",
            "exclusion_reason"
        ]
    ].head(10).to_string(index=False)
)

print("\nIncluded examples:")
print(
    result.loc[
        result["include"] == 1,
        [
            "line_id",
            "line_description",
            "gl_account_description",
            "include"
        ]
    ].head(10).to_string(index=False)
)

print("\nMissing spend among included records:")
print(
    result.loc[result["include"] == 1, "spend_usd"]
    .isna()
    .sum()
)

print("\nValidation completed.")

