import re

import pandas as pd

from ..preprocessing.normalization import normalize_text

NON_PROCUREMENT_TERMS = (
    "intercompany", "inter company", "interco", "royalty", "royalties", "income tax",
    "vat payment", "withholding tax", "corporate tax", "financing cost", "interest charge",
    "loan repayment", "partner billing", "partner bill", "revenue share", "dividend",
)


def non_procurement_reason(values) -> str:
    text = normalize_text(" ".join(str(v) for v in values if pd.notna(v)))
    for term in NON_PROCUREMENT_TERMS:
        if normalize_text(term) in text:
            return "NON_PROCUREMENT"
    # The account descriptions are codes in several regions; these explicit variants signal IC.
    if re.search(r"\b(ic payable|i c payable|intercompany payable|intercompany sales)\b", text):
        return "NON_PROCUREMENT"
    return ""


def tool_status_reason(status) -> str:
    value = normalize_text(status)
    if "cancel" in value or value in {"void", "voided"}:
        return "CANCELLED"
    if value not in {"approved", "paid"}:
        return "NOT_APPROVED"
    return ""
