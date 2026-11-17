from .inclusion_rules import non_procurement_reason, tool_status_reason


def decide_exclusion(kind: str, row, fields: dict) -> str:
    if kind == "tool":
        reason = tool_status_reason(row.get(fields.get("status", "status")))
        if reason:
            return reason
    text_columns = fields.get("non_procurement_text", [])
    values = [row.get(column) for column in text_columns]
    return non_procurement_reason(values)
