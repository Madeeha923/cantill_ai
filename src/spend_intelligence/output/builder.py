import pandas as pd


OUTPUT_COLUMNS = [
    "record_id",
    "include",
    "exclusion_reason",
    "supplier_key",
    "item_key",
    "spend",
    "l1",
    "l2",
    "l3",
    "l4",
    "confidence",
]


def build_output(df):
    result = df.copy()

    for column in OUTPUT_COLUMNS:
        if column not in result.columns:
            result[column] = ""

    result["include"] = (
        pd.to_numeric(result["include"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    result["spend"] = pd.to_numeric(
        result["spend"], errors="coerce"
    ).fillna(0.0)

    result.loc[result["include"] == 0, "spend"] = 0.0
    result.loc[result["include"] == 1, "exclusion_reason"] = ""

    result["confidence"] = (
        pd.to_numeric(result["confidence"], errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    for column in [
        "record_id", "exclusion_reason", "supplier_key",
        "item_key", "l1", "l2", "l3", "l4"
    ]:
        result[column] = result[column].fillna("").astype(str)

    result = result[OUTPUT_COLUMNS]

    if result["record_id"].duplicated().any():
        raise ValueError("Duplicate record_id values in output.")

    if not result["include"].isin([0, 1]).all():
        raise ValueError("Invalid include values.")

    return result
