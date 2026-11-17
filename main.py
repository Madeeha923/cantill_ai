
import argparse
from pathlib import Path

import pandas as pd

from src.spend_intelligence.config import load_config
from src.spend_intelligence.ingestion.loader import load_company_inputs
from src.spend_intelligence.preprocessing.cleaning import clean_text
from src.spend_intelligence.preprocessing.company_a import prepare_company_a
from src.spend_intelligence.preprocessing.company_b import prepare_company_b, apply_company_b_inclusion
from src.spend_intelligence.preprocessing.identity import (
    resolve_supplier_key,
    resolve_item_key,
)
from src.spend_intelligence.preprocessing.classification import TaxonomyClassifier
from src.spend_intelligence.output.builder import build_output
from src.spend_intelligence.entity_resolution.deduplication import (
    duplicate_tool_rows,
    match_a_tool_to_erp_diagnostics,
)
from src.spend_intelligence.spend.inclusion_rules import tool_status_reason
from src.spend_intelligence.pipeline import run_generic_pipeline


def make_text(df, columns):
    available = [c for c in columns if c in df.columns]
    return (
        df[available].fillna("").astype(str)
        .agg(" ".join, axis=1).str.strip()
    )


def company_a_tool_exclusion_reason(status, matched=False, repeated=False):
    """Apply required status precedence, then tool duplicate exclusions."""
    reason = tool_status_reason(status)
    if reason:
        return reason
    if matched or repeated:
        return "DUPLICATE"
    return ""


def classify_records(records, training, taxonomy, text_column):
    classifier = TaxonomyClassifier()

    classifier.fit(
        training,
        [text_column],
        ["category_l1", "category_l2", "category_l3"]
    )

    return classifier.predict(
        records,
        [text_column],
        taxonomy
    )


def run_company_a(config, inputs):
    data = prepare_company_a(inputs, config)

    erp = data["erp"].copy()
    tool = data["invoice_tool"].copy()
    labels = inputs["labels"]
    taxonomy = inputs["taxonomy"]

    # Train using ERP records joined to golden labels.
    erp["record_id"] = "ERP:" + erp["line_id"].astype(str)
    training = erp.merge(labels, on="record_id", how="inner")

    erp["classification_text"] = make_text(
        erp,
        [
            "material_short_text", "service_short_text",
            "material_number", "spend_type", "material_type"
        ]
    )

    training["classification_text"] = make_text(
        training,
        [
            "material_short_text", "service_short_text",
            "material_number", "spend_type", "material_type"
        ]
    )

    # Classify ERP and Invoice Tool records using the same model.
    tool["classification_text"] = make_text(
        tool, ["line_text"]
    )

    all_records = pd.concat(
        [
            erp.assign(source="ERP"),
            tool.assign(source="TOOL")
        ],
        ignore_index=True,
        sort=False
    )

    classified = classify_records(
        all_records,
        training,
        taxonomy,
        "classification_text"
    )

    # Restore golden supplier/item keys for ERP records.
    golden_keys = (
        labels[["record_id", "supplier_key", "item_key"]]
        .drop_duplicates("record_id")
        .rename(columns={
            "supplier_key": "supplier_key_gold",
            "item_key": "item_key_gold",
        })
    )

    erp_mask = classified["source"] == "ERP"
    classified.loc[erp_mask, "record_id"] = (
        "ERP:" + classified.loc[erp_mask, "line_id"].astype(str)
    )

    classified = classified.merge(
        golden_keys,
        on="record_id",
        how="left"
    )

    classified["supplier_key"] = classified["supplier_key_gold"]
    classified["item_key"] = classified["item_key_gold"]

    # Fallback identity keys for records without golden keys.
    classified["supplier_key"] = classified["supplier_key"].fillna(
        classified.get("supplier_name_as_captured")
    )
    classified["item_key"] = classified["item_key"].fillna(
        classified.get("line_text")
    )

    classified["supplier_key"] = classified["supplier_key"].apply(
        lambda x: str(x).strip() if pd.notna(x) else ""
    )
    classified["item_key"] = classified["item_key"].apply(
        lambda x: str(x).strip() if pd.notna(x) else ""
    )

    classified["include"] = 1
    classified["exclusion_reason"] = ""

    # Conservative cross-source deduplication. ERP rows remain canonical;
    # ambiguous PO/item/supplier/amount matches are not excluded.
    erp_keys = erp.copy()
    erp_keys["po"] = erp_keys["po_number"].astype("string").str.strip()
    erp_keys["po_item_key"] = (
        pd.to_numeric(erp_keys["po_item"], errors="coerce")
        .astype("Int64").astype("string")
    )
    erp_keys["supplier_norm"] = (
        erp_keys["supplier_name"].fillna("").str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )

    tool_keys = tool.copy()
    tool_keys["po"] = (
        tool_keys["po_reference"].astype("string")
        .str.extract(r"(\d{8,})", expand=False)
    )
    tool_keys["po_item_key"] = (
        tool_keys["po_reference"].astype("string")
        .str.extract(r"/\s*(\d+(?:\.0)?)", expand=False)
    )
    tool_keys["po_item_key"] = (
        pd.to_numeric(tool_keys["po_item_key"], errors="coerce")
        .astype("Int64").astype("string")
    )
    tool_keys["supplier_norm"] = (
        tool_keys["supplier_name_as_captured"].fillna("").str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )

    match_fields = {
        "erp_po": "po",
        "tool_po": "po",
        "erp_item": "po_item_key",
        "tool_item": "po_item_key",
        "erp_supplier": "supplier_norm",
        "tool_supplier": "supplier_norm",
        "erp_amount": "spend_zar",
        "tool_amount": "spend_zar",
        "tool_amount_is_net": True,
        "tool_description": "line_text",
        "erp_description": ["material_short_text", "service_short_text"],
    }
    matches, ambiguous_indices = match_a_tool_to_erp_diagnostics(
        erp_keys, tool_keys, match_fields
    )
    repeated_indices = duplicate_tool_rows(
        tool_keys,
        {
            "tool_invoice": "invoice_number",
            "tool_supplier": "supplier_norm",
            "tool_po": "po",
            "tool_item": "po_item_key",
            "tool_description": "line_text",
            "tool_amount": "spend_zar",
        },
    )

    tool_mask = classified["source"] == "TOOL"
    status_reasons = {}
    for tool_idx, row in tool_keys.iterrows():
        status_reasons[str(row["tool_record_id"])] = company_a_tool_exclusion_reason(
            row.get("status", ""),
            matched=tool_idx in matches,
            repeated=tool_idx in repeated_indices,
        )
    classified.loc[tool_mask, "exclusion_reason"] = (
        classified.loc[tool_mask, "tool_record_id"].astype(str).map(status_reasons)
    )
    classified.loc[tool_mask & classified["exclusion_reason"].ne(""), "include"] = 0

    classified["record_id"] = classified.apply(
        lambda r: (
            "ERP:" + str(r["line_id"])
            if r["source"] == "ERP"
            else "TOOL:" + str(r["tool_record_id"])
        ),
        axis=1
    )

    classified["spend"] = classified["spend_zar"]

    match_diagnostic_rows = []
    for tool_idx, row in tool_keys.iterrows():
        if tool_idx in repeated_indices:
            match_classification = "repeated"
        elif tool_idx in matches:
            match_classification = "matched"
        elif tool_idx in ambiguous_indices:
            match_classification = "ambiguous"
        else:
            match_classification = "unmatched"
        erp_idx = matches.get(tool_idx)
        match_diagnostic_rows.append({
            "record_id": "TOOL:" + str(row["tool_record_id"]),
            "status": row.get("status", ""),
            "status_exclusion_reason": status_reasons.get(str(row["tool_record_id"]), ""),
            "match_classification": match_classification,
            "matched_erp_record_id": (
                "ERP:" + str(erp.loc[erp_idx, "line_id"])
                if erp_idx is not None else ""
            ),
        })

    diagnostics = pd.DataFrame(match_diagnostic_rows)
    status_summary = (
        classified.loc[tool_mask]
        .groupby("status", dropna=False)
        .agg(records=("include", "size"), included=("include", "sum"))
        .reset_index()
    )
    status_summary["excluded"] = status_summary["records"] - status_summary["included"]
    classified.attrs["company_a_matching_diagnostics"] = diagnostics
    classified.attrs["company_a_status_counts"] = status_summary
    return classified


def run_company_b(config, inputs):
    data = prepare_company_b(inputs, config)
    records = apply_company_b_inclusion(data["invoice_lines"])

    labels = inputs["labels"]
    taxonomy = inputs["taxonomy"]

    records["classification_text"] = make_text(
        records,
        [
            "line_description", "material_number",
            "gl_account_description", "supplier_type"
        ]
    )

    training = records.merge(
        labels[["line_id", "category_l1", "category_l2", "category_l3"]],
        on="line_id",
        how="inner",
        suffixes=("", "_label")
    )

    for level in ["category_l1", "category_l2", "category_l3"]:
        label_col = f"{level}_label"
        if label_col in training.columns:
            training[level] = training[label_col]

    training = training.dropna(
        subset=["category_l1", "category_l2", "category_l3"]
    )

    classified = classify_records(
        records,
        training,
        taxonomy,
        "classification_text"
    )

    classified = resolve_supplier_key(
        classified, "vendor_id", "vendor_id"
    )
    classified = resolve_item_key(
        classified, "material_number", "line_description"
    )

    classified["record_id"] = classified["line_id"].astype(str)
    classified["spend"] = classified["spend_usd"]

    return classified


def main():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--company", choices=["A", "B", "C"])
    selection.add_argument("--config", help="Path to a company YAML configuration.")
    args = parser.parse_args()

    config_path = (
        Path(args.config)
        if args.config
        else Path("configs") / f"company_{args.company.lower()}.yaml"
    )
    config = load_config(config_path)
    company = str(config["company"]).strip().lower()
    if args.company and company != args.company.lower():
        parser.error(
            f"Selected company {args.company} does not match config company {config['company']}."
        )

    if config.get("template_only"):
        parser.error(
            f"{config_path} is a template. Replace its file paths and column mappings "
            "when Company C's blind data arrives; no actual C schema is assumed."
        )

    if company == "a":
        inputs = load_company_inputs(config)
        result = run_company_a(config, inputs)
    elif company == "b":
        inputs = load_company_inputs(config)
        result = run_company_b(config, inputs)
    elif config.get("mode") == "generic":
        result = run_generic_pipeline(config, project_root=Path.cwd())
    else:
        parser.error(
            f"Config for company '{config['company']}' must set mode: generic."
        )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    if company == "a":
        status_counts = result.attrs.get("company_a_status_counts")
        if status_counts is not None:
            print("\nCompany A Invoice Tool inclusion by status:")
            print(status_counts.to_string(index=False))
        match_diagnostics = result.attrs.get("company_a_matching_diagnostics")
        if match_diagnostics is not None:
            diagnostic_path = output_dir / "company_a_matching_diagnostics.csv"
            match_diagnostics.to_csv(diagnostic_path, index=False)
            print("\nCompany A tool matching diagnostics:")
            print(match_diagnostics["match_classification"].value_counts().to_string())
            print("Diagnostics:", diagnostic_path)
    elif company == "b":
        missing = result.loc[result["spend_missing"]].copy()
        if not missing.empty:
            missing_diagnostics = pd.DataFrame({
                "record_id": missing["line_id"].astype(str),
                "currency": missing["currency"].astype(str),
                "local_amount": missing["local_amount"],
                "fx_source": missing["fx_source"],
                "missing_fx_rate": missing["fx_missing"],
                "missing_amount": missing["amount_missing"],
            })
            missing_diagnostics["missing_reason"] = missing_diagnostics.apply(
                lambda row: ";".join(
                    reason for flag, reason in (
                        (row["missing_amount"], "MISSING_AMOUNT"),
                        (row["missing_fx_rate"], "MISSING_FX_RATE"),
                    ) if flag
                ),
                axis=1,
            )
        else:
            missing_diagnostics = pd.DataFrame(columns=[
                "record_id", "currency", "local_amount", "fx_source",
                "missing_fx_rate", "missing_amount", "missing_reason",
            ])
        diagnostic_path = output_dir / "company_b_missing_spend.csv"
        missing_diagnostics.to_csv(diagnostic_path, index=False)
        print("\nCompany B missing-spend diagnostics:")
        if missing_diagnostics.empty:
            print("No missing spend records.")
        else:
            print(missing_diagnostics.to_string(index=False))
        print("Diagnostics:", diagnostic_path)
    else:
        missing = result.attrs.get("generic_missing_spend", pd.DataFrame())
        if not missing.empty:
            diagnostic_path = output_dir / f"company_{company}_missing_spend.csv"
            missing.to_csv(diagnostic_path, index=False)
            print("Generic missing-spend diagnostics:", diagnostic_path)

    output = build_output(result)

    output_path = output_dir / f"company_{company}_output.csv"
    output.to_csv(output_path, index=False)

    print("\nPipeline completed.")
    print("Output:", output_path)
    print("Rows:", len(output))
    print("Included:", (output["include"] == 1).sum())
    print("Excluded:", (output["include"] == 0).sum())


if __name__ == "__main__":
    main()
