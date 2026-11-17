import re
import unicodedata


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def stable_key(prefix: str, value: object) -> str:
    normalized = normalize_text(value)
    if not normalized:
        return f"{prefix}-unknown"
    return f"{prefix}-" + re.sub(r"\s+", "-", normalized)[:80]


def compact_id(value: object) -> str:
    return re.sub(r"\D", "", str(value or ""))
