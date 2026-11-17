from ..preprocessing.normalization import stable_key


def supplier_key(name=None, vendor_id=None) -> str:
    if vendor_id is not None and str(vendor_id).strip() not in {"", "nan", "None"}:
        return stable_key("supplier", vendor_id)
    return stable_key("supplier", name)


def apply_label_mapping(value, mapping, fallback):
    if value is None:
        return fallback
    return mapping.get(str(value), fallback)
