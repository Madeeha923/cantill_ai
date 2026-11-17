import re
import pandas as pd


def normalize_key(value):
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in {"", "nan", "none", "null", "unknown", "n/a"}:
        return None

    return re.sub(r"[^a-z0-9]", "", value)


def resolve_supplier_key(df, id_column, name_column):
    result = df.copy()

    supplier_id = result[id_column].apply(normalize_key)
    supplier_name = result[name_column].apply(normalize_key)

    result["supplier_key"] = supplier_id

    result.loc[
        result["supplier_key"].isna(),
        "supplier_key"
    ] = supplier_name

    return result


def resolve_item_key(df, id_column, description_column):
    result = df.copy()

    item_id = result[id_column].apply(normalize_key)
    description = result[description_column].apply(normalize_key)

    result["item_key"] = item_id

    result.loc[
        result["item_key"].isna(),
        "item_key"
    ] = description

    return result