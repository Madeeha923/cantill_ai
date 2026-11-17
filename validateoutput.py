
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent

OUTPUT_COLUMNS = [
    "record_id", "include", "exclusion_reason", "supplier_key",
    "item_key", "spend", "l1", "l2", "l3", "l4", "confidence"
]


def validate_company(company):
    output_path = ROOT / f"outputs/company_{company.lower()}_output.csv"
    taxonomy_path = ROOT / f"data/company_{company.lower()}/taxonomy_{company}.csv"

    df = pd.read_csv(output_path)
    taxonomy = pd.read_csv(taxonomy_path).fillna("")

    print(f"\n{'=' * 15} COMPANY {company} {'=' * 15}")

    # Required columns
    missing_columns = set(OUTPUT_COLUMNS) - set(df.columns)
    print("Missing columns:", missing_columns)

    # Row and ID checks
    print("Total rows:", len(df))
    print("Duplicate IDs:", df["record_id"].duplicated().sum())
    print("Missing IDs:", df["record_id"].isna().sum())

    # Inclusion and spend checks
    print("Invalid include values:", (~df["include"].isin([0, 1])).sum())
    print("Excluded rows with non-zero spend:",
          ((df["include"] == 0) & (df["spend"] != 0)).sum())
    print("Invalid confidence values:",
          ((df["confidence"] < 0) | (df["confidence"] > 1)).sum())

    # Taxonomy validation
    taxonomy_paths = set(
        taxonomy[["category_l1", "category_l2", "category_l3"]]
        .astype(str)
        .apply(tuple, axis=1)
    )

    output_paths = df[["l1", "l2", "l3"]].fillna("").astype(str)
    invalid_paths = ~output_paths.apply(tuple, axis=1).isin(taxonomy_paths)

    nonempty_paths = output_paths.ne("").any(axis=1)
    print("Non-empty category paths:", nonempty_paths.sum())
    print("Invalid taxonomy paths:", (invalid_paths & nonempty_paths).sum())

    print("Included:", (df["include"] == 1).sum())
    print("Excluded:", (df["include"] == 0).sum())


def validate_company_b_labels():
    output = pd.read_csv(ROOT / "outputs/company_b_output.csv")
    labels = pd.read_csv(ROOT / "data/company_b/seed_labels.csv")

    labels = labels.rename(columns={"include": "actual_include"})
    predictions = output[["record_id", "include"]].rename(
        columns={"include": "predicted_include"}
    )

    merged = labels.merge(
        predictions,
        left_on="line_id",
        right_on="record_id",
        how="left"
    )

    print("\n========== COMPANY B SEED LABEL VALIDATION ==========")
    print("Seed rows:", len(merged))
    print("Missing predictions:", merged["predicted_include"].isna().sum())
    scored = merged["predicted_include"].notna()
    accuracy = (
        merged.loc[scored, "actual_include"]
        == merged.loc[scored, "predicted_include"]
    ).mean()
    print("Inclusion accuracy:", accuracy)

    print("\nConfusion matrix:")
    print(pd.crosstab(
        merged["actual_include"],
        merged["predicted_include"],
        rownames=["Actual"],
        colnames=["Predicted"]
    ))


if __name__ == "__main__":
    validate_company("A")
    validate_company("B")
    validate_company_b_labels()
