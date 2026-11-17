
import pandas as pd

from src.spend_intelligence.config import load_config
from src.spend_intelligence.ingestion.loader import load_company_inputs
from src.spend_intelligence.preprocessing.company_b import prepare_company_b


config = load_config("configs/company_b.yaml")
inputs = load_company_inputs(config)

processed = prepare_company_b(inputs, config)
df = processed["invoice_lines"].copy()
labels = inputs["labels"].copy()

# Attach seed labels to invoice records.
df = df.merge(
    labels,
    on="line_id",
    how="left",
    indicator=True,
    suffixes=("", "_label")
)

df["labelled"] = df["_merge"].eq("both")

print("\n========== LABEL COVERAGE ==========")
print(f"Total invoice lines: {len(df):,}")
print(f"Labelled lines: {df['labelled'].sum():,}")
print(f"Unlabelled lines: {(~df['labelled']).sum():,}")

labelled = df[df["labelled"]].copy()

# 1. Overall inclusion/exclusion distribution.
print("\n========== OVERALL INCLUSION ==========")
print(labelled["include"].value_counts(dropna=False))

# 2. Exclusion reasons.
print("\n========== EXCLUSION REASONS ==========")
print(
    labelled.loc[labelled["include"] == 0, "exclusion_reason"]
    .value_counts(dropna=False)
)

# 3. Inclusion by document type.
print("\n========== DOCUMENT TYPE ANALYSIS ==========")
doc_analysis = (
    labelled.groupby("document_type", dropna=False)
    .agg(
        labelled_rows=("line_id", "count"),
        included=("include", lambda x: (x == 1).sum()),
        excluded=("include", lambda x: (x == 0).sum()),
    )
)

doc_analysis["inclusion_rate"] = (
    doc_analysis["included"] / doc_analysis["labelled_rows"]
)

print(doc_analysis.sort_values("labelled_rows", ascending=False))

# 4. Exclusion reason by document type.
print("\n========== EXCLUSION REASON BY DOCUMENT TYPE ==========")
excluded = labelled[labelled["include"] == 0]

print(
    pd.crosstab(
        excluded["document_type"],
        excluded["exclusion_reason"],
        margins=True
    )
)

# 5. Inclusion by source system.
print("\n========== SOURCE SYSTEM ANALYSIS ==========")
source_analysis = (
    labelled.groupby("source_system", dropna=False)
    .agg(
        labelled_rows=("line_id", "count"),
        included=("include", lambda x: (x == 1).sum()),
        excluded=("include", lambda x: (x == 0).sum()),
    )
)

source_analysis["inclusion_rate"] = (
    source_analysis["included"] / source_analysis["labelled_rows"]
)

print(source_analysis.sort_values("labelled_rows", ascending=False))

# 6. Inclusion by region.
print("\n========== REGION ANALYSIS ==========")
region_analysis = (
    labelled.groupby("region", dropna=False)
    .agg(
        labelled_rows=("line_id", "count"),
        included=("include", lambda x: (x == 1).sum()),
        excluded=("include", lambda x: (x == 0).sum()),
    )
)

region_analysis["inclusion_rate"] = (
    region_analysis["included"] / region_analysis["labelled_rows"]
)

print(region_analysis.sort_values("labelled_rows", ascending=False))

# 7. Inspect all distinct exclusion reasons.
print("\n========== DISTINCT EXCLUSION REASONS ==========")
print(sorted(
    excluded["exclusion_reason"]
    .dropna()
    .astype(str)
    .unique()
))

print("\nAnalysis completed.")



import pandas as pd

from src.spend_intelligence.config import load_config
from src.spend_intelligence.ingestion.loader import load_company_inputs


config = load_config("configs/company_b.yaml")
inputs = load_company_inputs(config)

df = inputs["invoice_lines"].merge(
    inputs["labels"],
    on="line_id",
    how="inner"
)

# Compare included and excluded records.
fields = [
    "document_type",
    "source_system",
    "region",
    "supplier_type",
    "gl_account",
    "gl_account_description",
    "cost_centre",
    "po_type",
    "transport_flag",
]

for column in fields:
    print(f"\n========== {column.upper()} ==========")

    summary = (
        df.groupby(
            [column, "include"],
            dropna=False
        )
        .size()
        .unstack(fill_value=0)
    )

    summary.columns = [
        f"include_{int(c)}" for c in summary.columns
    ]

    summary["total"] = summary.sum(axis=1)

    print(
        summary.sort_values(
            "total",
            ascending=False
        ).head(25).to_string()
    )


# Compare text descriptions.
print("\n========== EXAMPLE INCLUDED DESCRIPTIONS ==========")
print(
    df.loc[
        df["include"] == 1,
        ["line_description", "gl_account_description"]
    ]
    .dropna(subset=["line_description"])
    .head(30)
    .to_string(index=False)
)

print("\n========== EXAMPLE EXCLUDED DESCRIPTIONS ==========")
print(
    df.loc[
        df["include"] == 0,
        ["line_description", "gl_account_description"]
    ]
    .dropna(subset=["line_description"])
    .head(50)
    .to_string(index=False)
)


import pandas as pd

from src.spend_intelligence.config import load_config
from src.spend_intelligence.ingestion.loader import load_company_inputs


config = load_config("configs/company_b.yaml")
inputs = load_company_inputs(config)

all_lines = inputs["invoice_lines"].copy()
labels = inputs["labels"].copy()

df = all_lines.merge(
    labels,
    on="line_id",
    how="left",
    indicator=True
)

labelled = df[df["_merge"] == "both"].copy()

# Signals worth investigating.
fields = [
    "supplier_type",
    "gl_account",
    "gl_account_description",
    "document_type",
    "po_type",
]

for column in fields:
    print(f"\n========== {column.upper()} ==========")

    summary = (
        labelled.groupby(column, dropna=False)
        .agg(
            labelled=("line_id", "count"),
            excluded=("include", lambda x: (x == 0).sum()),
            included=("include", lambda x: (x == 1).sum()),
        )
    )

    summary["exclusion_rate"] = (
        summary["excluded"] / summary["labelled"]
    )

    print(
        summary.sort_values(
            ["exclusion_rate", "labelled"],
            ascending=[False, False]
        ).to_string()
    )


# Check whether promising supplier types and GL accounts
# also occur frequently among unlabelled records.
print("\n========== FULL DATA: SUPPLIER TYPES ==========")

for value in ["5406", "2742"]:
    subset = all_lines[
        all_lines["supplier_type"].astype("string").str.strip() == value
    ]
    print(f"{value}: {len(subset):,} total records")


print("\n========== FULL DATA: GL ACCOUNTS ==========")

for value in ["GL40242", "GL40261", "GL40102", "GL40125"]:
    subset = all_lines[
        all_lines["gl_account"].astype("string").str.strip() == value
    ]
    print(f"{value}: {len(subset):,} total records")


# Inspect the complete descriptions associated with exclusions.
print("\n========== EXCLUDED GL DESCRIPTIONS ==========")

excluded = labelled[labelled["include"] == 0]

print(
    excluded.groupby(
        ["gl_account", "gl_account_description"],
        dropna=False
    )
    .size()
    .sort_values(ascending=False)
    .to_string()
)


# Inspect examples of excluded and included transactions
# for the strongest candidate GL accounts.
print("\n========== GL40242 / GL40261 EXAMPLES ==========")

examples = labelled[
    labelled["gl_account"].isin(["GL40242", "GL40261"])
]

columns = [
    "line_id",
    "source_system",
    "supplier_type",
    "gl_account",
    "gl_account_description",
    "line_description",
    "include",
    "exclusion_reason",
]

print(examples[columns].to_string(index=False))

print("\nValidation completed.")



import pandas as pd

from src.spend_intelligence.config import load_config
from src.spend_intelligence.ingestion.loader import load_company_inputs


config = load_config("configs/company_b.yaml")
inputs = load_company_inputs(config)

df = inputs["invoice_lines"].copy()
labels = inputs["labels"].copy()

df = df.merge(
    labels[["line_id", "include"]],
    on="line_id",
    how="left"
)

df["gl_account_description"] = (
    df["gl_account_description"]
    .astype("string")
    .str.strip()
)

df["line_description"] = (
    df["line_description"]
    .astype("string")
    .str.strip()
)

print("\n========== INTERCOMPANY GL DESCRIPTIONS ==========")

ic_mask = df["gl_account_description"].str.contains(
    r"^I\s*/?\s*C|^IC",
    case=False,
    na=False,
    regex=True
)

ic = df[ic_mask]

print("Total matching records:", len(ic))
print("Labelled:", ic["include"].notna().sum())
print("Unlabelled:", ic["include"].isna().sum())
print("\nGL description distribution:")
print(ic["gl_account_description"].value_counts().to_string())

print("\n========== GL40242 AND GL40261 ==========")

for account in ["GL40242", "GL40261"]:
    subset = df[df["gl_account"] == account]

    print(f"\n{account}")
    print("Total:", len(subset))
    print("Labelled:", subset["include"].notna().sum())
    print("Unlabelled:", subset["include"].isna().sum())
    print("Descriptions:")
    print(subset["line_description"].value_counts().head(15).to_string())

print("\n========== SUPPLIER TYPES ==========")

for supplier_type in ["5406", "2742"]:
    subset = df[
        df["supplier_type"].astype("string").str.strip()
        == supplier_type
    ]

    print(f"\nSupplier type {supplier_type}")
    print("Total:", len(subset))
    print("Labelled:", subset["include"].notna().sum())
    print("Unlabelled:", subset["include"].isna().sum())
    print("GL descriptions:")
    print(
        subset["gl_account_description"]
        .value_counts(dropna=False)
        .head(10)
        .to_string()
    )

print("\n========== SALES TAX DESCRIPTIONS ==========")

tax_mask = df["line_description"].str.contains(
    r"\bsales\s+tax\b",
    case=False,
    na=False,
    regex=True
)

tax = df[tax_mask]

print("Total matching records:", len(tax))
print("Labelled:", tax["include"].notna().sum())
print("Unlabelled:", tax["include"].isna().sum())
print("\nLabel distribution:")
print(tax["include"].value_counts(dropna=False).to_string())

print("\nValidation completed.")
