import pandas as pd

from ..preprocessing.cleaning import combine_fields


def make_text(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return pd.Series([""] * len(frame), index=frame.index, dtype="string")
    return frame.apply(lambda row: combine_fields(row, existing), axis=1).astype("string").fillna("")
