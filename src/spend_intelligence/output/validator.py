import pandas as pd

REQUIRED_COLUMNS = ["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend",
                    "l1", "l2", "l3", "l4", "confidence"]
VALID_REASONS = {"", "CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE"}


def validate_output(frame: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Output missing required columns: {sorted(missing)}")
    if frame["record_id"].duplicated().any():
        raise ValueError("Output has duplicate record_id values")
    if not frame["include"].isin([0, 1]).all():
        raise ValueError("include must contain only 0 or 1")
    if not frame["exclusion_reason"].fillna("").isin(VALID_REASONS).all():
        raise ValueError("Output contains an unknown exclusion reason")
    if frame.loc[frame["include"].eq(0), "exclusion_reason"].fillna("").eq("").any():
        raise ValueError("Every excluded row needs an exclusion reason")
    if not frame["confidence"].between(0, 1).all():
        raise ValueError("Confidence must be between zero and one")
    if not frame["spend"].map(lambda x: isinstance(x, (int, float))).all():
        raise ValueError("Spend must be numeric")
