"""Config-driven generic pipeline for companies without training labels.

Company A and B keep their existing specialized implementations in main.py;
this module is the single generic branch called by that CLI dispatcher.
"""

from pathlib import Path

import pandas as pd

from .entity_resolution.item_matching import item_key as make_item_key
from .entity_resolution.supplier_matching import supplier_key as make_supplier_key
from .ingestion.loader import load_sources
from .ingestion.schema import require_columns
from .output.builder import build_output
from .preprocessing.classification import TaxonomyClassifier
from .preprocessing.cleaning import parse_numeric


VALID_EXCLUSION_REASONS = {
    "CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE"
}


def _column_name(value):
    if value is None or not str(value).strip():
        return None
    return str(value).strip().lower()


def _configured_value(row, column):
    if not column:
        return None
    value = row.get(column)
    return None if pd.isna(value) else value


def _monthly_fx_lookup(fx_frame, fx_config):
    if not fx_config:
        return {}
    currency_column = _column_name(fx_config.get("currency_column"))
    month_column = _column_name(fx_config.get("month_column"))
    rate_column = _column_name(fx_config.get("rate_column"))
    missing_settings = [
        name for name, value in (
            ("currency_column", currency_column),
            ("month_column", month_column),
            ("rate_column", rate_column),
        ) if not value
    ]
    if missing_settings:
        raise ValueError(f"generic.monthly_fx is missing mappings: {missing_settings}")
    require_columns(
        fx_frame,
        [currency_column, month_column, rate_column],
        "monthly FX source",
    )
    rates = fx_frame[[currency_column, month_column, rate_column]].copy()
    rates["_currency"] = rates[currency_column].astype("string").str.strip().str.upper()
    rates["_month"] = rates[month_column].astype("string").str.strip().str[:7]
    rates["_rate"] = pd.to_numeric(rates[rate_column], errors="coerce")
    rates = rates.loc[rates["_rate"].gt(0)]
    if rates.duplicated(["_currency", "_month"]).any():
        raise ValueError("Monthly FX source has duplicate currency-month rates.")
    return {
        (row["_currency"], row["_month"]): float(row["_rate"])
        for _, row in rates.iterrows()
    }


def run_generic_pipeline(config: dict, project_root: str | Path = ".") -> pd.DataFrame:
    """Run a no-label, taxonomy-driven pipeline from a generic YAML mapping."""
    if config.get("mode") != "generic":
        raise ValueError("Generic pipeline requires mode: generic in the YAML config.")

    files = config.get("files", {})
    if not files.get("transactions") or not files.get("taxonomy"):
        raise ValueError("Generic config files must define transactions and taxonomy paths.")

    fields = config.get("generic")
    if not isinstance(fields, dict):
        raise ValueError("Generic config must contain a 'generic' field mapping.")

    record_id_column = _column_name(fields.get("record_id_column"))
    amount_column = _column_name(fields.get("amount_column"))
    currency_column = _column_name(fields.get("currency_column"))
    date_column = _column_name(fields.get("date_column"))
    rate_column = _column_name(fields.get("transaction_rate_column"))
    supplier_id_column = _column_name(fields.get("supplier_id_column"))
    supplier_name_column = _column_name(fields.get("supplier_name_column"))
    item_id_column = _column_name(fields.get("item_id_column"))
    item_description_column = _column_name(fields.get("item_description_column"))
    text_columns = [_column_name(column) for column in fields.get("text_columns", [])]
    text_columns = [column for column in text_columns if column]
    taxonomy_columns = [_column_name(column) for column in fields.get("taxonomy_columns", [])]

    required_settings = {
        "record_id_column": record_id_column,
        "amount_column": amount_column,
        "currency_column": currency_column,
    }
    missing_settings = [name for name, value in required_settings.items() if not value]
    if missing_settings:
        raise ValueError(f"Generic config is missing mappings: {missing_settings}")
    if not text_columns:
        raise ValueError("Generic config text_columns must contain at least one column.")
    if len(taxonomy_columns) not in (3, 4) or any(not col for col in taxonomy_columns):
        raise ValueError("Generic config taxonomy_columns must list three or four columns.")
    if len(set(taxonomy_columns)) != len(taxonomy_columns):
        raise ValueError("Generic config taxonomy_columns must be unique.")

    default_include = fields.get("default_include", 1)
    if default_include not in (0, 1):
        raise ValueError("generic.default_include must be 0 or 1.")
    exclusion_reason = str(fields.get("default_exclusion_reason", "")).strip().upper()
    if default_include == 0 and exclusion_reason not in VALID_EXCLUSION_REASONS:
        raise ValueError(
            "When generic.default_include is 0, set a permitted default_exclusion_reason."
        )
    if default_include == 1:
        exclusion_reason = ""

    base_currency = str(config.get("base_currency", "")).strip().upper()
    if not base_currency:
        raise ValueError("Generic config must define base_currency.")

    frames = load_sources(config, project_root)
    transactions = frames["transactions"].copy()
    taxonomy = frames["taxonomy"].copy()

    transaction_columns = [
        record_id_column, amount_column, currency_column, *text_columns,
        date_column, rate_column, supplier_id_column, supplier_name_column,
        item_id_column, item_description_column,
    ]
    transaction_columns = list(dict.fromkeys(
        column for column in transaction_columns if column
    ))
    require_columns(transactions, transaction_columns, "transactions source")
    require_columns(taxonomy, taxonomy_columns, "taxonomy source")

    ids = transactions[record_id_column].astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any():
        raise ValueError(f"Records have missing values in configured ID column '{record_id_column}'.")
    if ids.duplicated().any():
        raise ValueError(f"Configured ID column '{record_id_column}' contains duplicate values.")

    taxonomy = taxonomy[taxonomy_columns].fillna("").copy()
    for column in taxonomy_columns:
        taxonomy[column] = taxonomy[column].astype("string").str.strip().fillna("")

    # Date parsing is for configured monthly FX lookup only; input records are retained.
    parsed_dates = None
    if date_column:
        parsed_dates = pd.to_datetime(
            transactions[date_column], format="mixed", errors="coerce"
        )

    amount_values, _ = parse_numeric(transactions[amount_column])
    currencies = transactions[currency_column].astype("string").str.strip().str.upper()
    transaction_rates = (
        pd.to_numeric(transactions[rate_column], errors="coerce")
        if rate_column else pd.Series(float("nan"), index=transactions.index)
    )

    monthly_fx_config = fields.get("monthly_fx") or {}
    fx_lookup = {}
    if monthly_fx_config:
        fx_source = monthly_fx_config.get("source")
        if not fx_source or fx_source not in frames:
            raise ValueError(
                "generic.monthly_fx.source must name a source defined under files."
            )
        if not date_column:
            raise ValueError("A date_column is required when monthly_fx is configured.")
        fx_lookup = _monthly_fx_lookup(frames[fx_source], monthly_fx_config)

    spends = []
    missing_spend_rows = []
    unresolved_fx_ids = []
    for position, (_, row) in enumerate(transactions.iterrows()):
        record_id = str(ids.iloc[position])
        local_amount = amount_values.iloc[position]
        currency = currencies.iloc[position]
        included = int(default_include)

        if not included:
            spends.append(0.0)
            continue

        if pd.isna(local_amount):
            spends.append(float("nan"))
            missing_spend_rows.append({
                "record_id": record_id,
                "reason": "MISSING_AMOUNT",
            })
            continue

        if pd.isna(currency) or not currency:
            raise ValueError(
                f"Record {record_id} has an amount but no currency in '{currency_column}'."
            )
        if currency == base_currency:
            spends.append(float(local_amount))
            continue

        rate = transaction_rates.iloc[position]
        if pd.isna(rate) or rate <= 0:
            rate = None
        if rate is None and fx_lookup and parsed_dates is not None:
            date_value = parsed_dates.iloc[position]
            if pd.notna(date_value):
                month = date_value.strftime("%Y-%m")
                rate = fx_lookup.get((currency, month))

        if rate is None or pd.isna(rate) or rate <= 0:
            unresolved_fx_ids.append(record_id)
            continue
        spends.append(float(local_amount) / float(rate))

    if unresolved_fx_ids:
        raise ValueError(
            "No valid transaction or monthly FX rate for non-base currency rows: "
            + ", ".join(unresolved_fx_ids[:10])
            + (f" (and {len(unresolved_fx_ids) - 10} more)" if len(unresolved_fx_ids) > 10 else "")
        )

    text_records = TaxonomyClassifier.taxonomy_fallback(
        transactions,
        text_columns,
        taxonomy,
        taxonomy_columns=taxonomy_columns,
    )

    suppliers = []
    items = []
    for _, row in transactions.iterrows():
        supplier_name = _configured_value(row, supplier_name_column)
        supplier_id = _configured_value(row, supplier_id_column)
        item_description = _configured_value(row, item_description_column)
        item_id = _configured_value(row, item_id_column)
        suppliers.append(make_supplier_key(supplier_name, supplier_id))
        items.append(make_item_key(item_description, item_id))

    result = pd.DataFrame({
        "record_id": ids.astype(str).to_numpy(),
        "include": int(default_include),
        "exclusion_reason": exclusion_reason,
        "supplier_key": suppliers,
        "item_key": items,
        "spend": spends,
        "l1": text_records["l1"].to_numpy(),
        "l2": text_records["l2"].to_numpy(),
        "l3": text_records["l3"].to_numpy(),
        "l4": text_records["l4"].to_numpy(),
        "confidence": text_records["confidence"].to_numpy(),
    })
    output = build_output(result)
    output.attrs["generic_missing_spend"] = pd.DataFrame(missing_spend_rows)
    return output
