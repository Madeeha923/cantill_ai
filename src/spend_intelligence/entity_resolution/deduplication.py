from collections import defaultdict
from difflib import SequenceMatcher

import pandas as pd

from ..preprocessing.normalization import compact_id, normalize_text


def _po_key(value):
    if value is None or pd.isna(value):
        return ""
    digits = compact_id(value)
    return digits[:10] if len(digits) >= 8 else ""


def _item_key(value):
    if value is None or pd.isna(value):
        return ""
    number = pd.to_numeric(value, errors="coerce")
    if pd.notna(number) and float(number).is_integer():
        return str(int(number))
    return normalize_text(value)


def match_a_tool_to_erp_diagnostics(
    erp: pd.DataFrame, tool: pd.DataFrame, fields: dict
) -> tuple[dict[int, int], set[int]]:
    """Match only on PO/item/supplier/amount evidence; report unresolved ties.

    When multiple ERP candidates have the same PO, item, and supplier,
    description similarity must select a unique candidate that also matches
    the amount. Equal or absent description evidence is ambiguous and is not
    matched; one amount match by itself is insufficient among multiple rows.
    ``tool_amount_is_net`` lets callers avoid converting already-net values.
    """
    erp_po = fields["erp_po"]
    tool_po = fields["tool_po"]
    erp_amount = fields["erp_amount"]
    tool_amount = fields["tool_amount"]
    erp_item = fields.get("erp_item")
    tool_item = fields.get("tool_item")
    erp_supplier = fields.get("erp_supplier")
    tool_supplier = fields.get("tool_supplier")
    description_fields = fields.get("erp_description", [])
    if isinstance(description_fields, str):
        description_fields = [description_fields]

    buckets = defaultdict(list)
    for erp_idx, row in erp.iterrows():
        po = _po_key(row.get(erp_po))
        item = _item_key(row.get(erp_item)) if erp_item else ""
        supplier = normalize_text(row.get(erp_supplier)) if erp_supplier else ""
        if po and (not erp_item or item) and (not erp_supplier or supplier):
            buckets[(po, item, supplier)].append(erp_idx)

    matches: dict[int, int] = {}
    ambiguous: set[int] = set()
    absolute_tolerance = float(fields.get("absolute_tolerance", 1.0))
    relative_tolerance = float(fields.get("relative_tolerance", 0.01))

    for tool_idx, row in tool.iterrows():
        po = _po_key(row.get(tool_po))
        item = _item_key(row.get(tool_item)) if tool_item else ""
        supplier = normalize_text(row.get(tool_supplier)) if tool_supplier else ""
        if not po or (tool_item and not item) or (tool_supplier and not supplier):
            continue

        candidates = buckets.get((po, item, supplier), [])
        amount = pd.to_numeric(row.get(tool_amount), errors="coerce")
        if pd.isna(amount):
            continue
        if not fields.get("tool_amount_is_net", False):
            tax = pd.to_numeric(row.get(fields.get("vat_rate")), errors="coerce")
            tax = 0.15 if pd.isna(tax) else tax
            amount = amount / (1 + tax)

        amount_matches = []
        for erp_idx in candidates:
            candidate_amount = pd.to_numeric(
                erp.loc[erp_idx].get(erp_amount), errors="coerce"
            )
            if pd.isna(candidate_amount):
                continue
            tolerance = max(
                absolute_tolerance,
                max(abs(float(amount)), abs(float(candidate_amount))) * relative_tolerance,
            )
            if abs(float(amount) - float(candidate_amount)) <= tolerance:
                amount_matches.append(erp_idx)

        if len(candidates) == 1 and len(amount_matches) == 1:
            matches[tool_idx] = amount_matches[0]
            continue
        if not amount_matches:
            continue

        tool_description = normalize_text(row.get(fields.get("tool_description"), ""))
        scores = []
        for erp_idx in candidates:
            erp_row = erp.loc[erp_idx]
            erp_description = normalize_text(
                " ".join(str(erp_row.get(column, "")) for column in description_fields)
            )
            score = (
                SequenceMatcher(None, tool_description, erp_description).ratio()
                if tool_description and erp_description
                else 0.0
            )
            scores.append((score, erp_idx))

        scores.sort(reverse=True)
        if (
            scores[0][0] > 0
            and scores[0][0] > scores[1][0]
            and scores[0][1] in amount_matches
        ):
            matches[tool_idx] = scores[0][1]
        else:
            ambiguous.add(tool_idx)

    return matches, ambiguous


def match_a_tool_to_erp(erp: pd.DataFrame, tool: pd.DataFrame, fields: dict) -> dict[int, int]:
    """Conservative PO-led match; returns tool row index -> ERP row index."""
    po_col = fields["erp_po"]
    tool_po_col = fields["tool_po"]
    value_col = fields["erp_amount"]
    tool_amount_col = fields["tool_amount"]
    tax_col = fields["vat_rate"]
    text_col = fields["tool_description"]
    erp_desc_cols = fields["erp_description"]
    buckets = defaultdict(list)
    for idx, value in erp[po_col].items():
        key = _po_key(value)
        if key:
            buckets[key].append(idx)
    used = set()
    matches = {}
    for ti, row in tool.iterrows():
        key = _po_key(row.get(tool_po_col))
        candidates = [ei for ei in buckets.get(key, []) if ei not in used]
        if not candidates:
            continue
        tax = pd.to_numeric(row.get(tax_col), errors="coerce")
        if pd.isna(tax):
            tax = 0.15
        gross = pd.to_numeric(row.get(tool_amount_col), errors="coerce")
        net = gross / (1 + tax) if pd.notna(gross) else None
        desc = normalize_text(row.get(text_col, ""))
        best_idx, best_score = None, -1.0
        for ei in candidates:
            erow = erp.loc[ei]
            amount = pd.to_numeric(erow.get(value_col), errors="coerce")
            amount_score = 0.0
            if net is not None and pd.notna(amount):
                amount_score = max(0.0, 1.0 - abs(float(net) - float(amount)) / max(abs(float(net)), abs(float(amount)), 1.0))
            erp_text = normalize_text(" ".join(str(erow.get(col, "")) for col in erp_desc_cols))
            text_score = SequenceMatcher(None, desc, erp_text).ratio() if desc and erp_text else 0.0
            # Same PO is strong evidence; amount/text break ties within a PO.
            score = 0.70 + 0.20 * amount_score + 0.10 * text_score
            if score > best_score:
                best_idx, best_score = ei, score
        if best_idx is not None:
            matches[ti] = best_idx
            used.add(best_idx)
    return matches


def duplicate_tool_rows(tool: pd.DataFrame, fields: dict) -> set[int]:
    cols = [fields.get("tool_invoice"), fields.get("tool_supplier"), fields.get("tool_po"),
            fields.get("tool_description"), fields.get("tool_amount"),
            fields.get("tool_item")]
    cols = [c for c in cols if c and c in tool.columns]
    signatures = set()
    duplicates = set()
    for idx, row in tool.iterrows():
        parts = []
        for col in cols:
            value = row.get(col)
            if col == fields.get("tool_amount"):
                number = pd.to_numeric(value, errors="coerce")
                parts.append(f"{number:.2f}" if pd.notna(number) else "")
            else:
                parts.append(normalize_text(value))
        signature = tuple(parts)
        if any(signature) and signature in signatures:
            duplicates.add(idx)
        else:
            signatures.add(signature)
    return duplicates
