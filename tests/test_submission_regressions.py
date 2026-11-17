import pandas as pd
import pytest

from main import company_a_tool_exclusion_reason
from src.spend_intelligence.entity_resolution.deduplication import (
    duplicate_tool_rows,
    match_a_tool_to_erp_diagnostics,
)
from src.spend_intelligence.output.builder import build_output
from src.spend_intelligence.preprocessing.company_b import prepare_company_b
from src.spend_intelligence.preprocessing.classification import TaxonomyClassifier
from src.spend_intelligence.spend.inclusion_rules import tool_status_reason


def _company_b_inputs(rows, fx_rows):
    return prepare_company_b(
        {
            "invoice_lines": pd.DataFrame(rows),
            "fx_rates": pd.DataFrame(fx_rows, columns=["currency", "month", "units_per_usd"]),
        },
        {
            "company_b": {
                "amount_column": "line_amount",
                "currency_column": "currency",
                "rate_column": "currency_rate",
                "fx_currency_column": "currency",
                "fx_month_column": "month",
                "fx_rate_column": "units_per_usd",
            }
        },
    )["invoice_lines"]


def test_company_a_status_rules_and_duplicate_reason_precedence():
    assert tool_status_reason("Approved") == ""
    assert tool_status_reason("Paid") == ""
    assert tool_status_reason("Cancelled") == "CANCELLED"
    assert tool_status_reason("On Hold") == "NOT_APPROVED"
    assert tool_status_reason("Rejected") == "NOT_APPROVED"

    assert company_a_tool_exclusion_reason("Approved", matched=True) == "DUPLICATE"
    assert company_a_tool_exclusion_reason("Paid", repeated=True) == "DUPLICATE"
    assert company_a_tool_exclusion_reason("Cancelled", matched=True) == "CANCELLED"
    assert company_a_tool_exclusion_reason("Rejected", repeated=True) == "NOT_APPROVED"
    assert company_a_tool_exclusion_reason("Approved") == ""


def test_company_a_repeated_tool_rows_need_matching_signature():
    tool = pd.DataFrame(
        [
            {"invoice": "INV-1", "supplier": "acme", "po": "12345678", "item": "10", "desc": "bolt", "amount": 100.0},
            {"invoice": "INV-1", "supplier": "acme", "po": "12345678", "item": "10", "desc": "bolt", "amount": 100.0},
            {"invoice": "INV-1", "supplier": "acme", "po": "12345678", "item": "20", "desc": "nut", "amount": 100.0},
        ]
    )
    repeated = duplicate_tool_rows(
        tool,
        {
            "tool_invoice": "invoice",
            "tool_supplier": "supplier",
            "tool_po": "po",
            "tool_item": "item",
            "tool_description": "desc",
            "tool_amount": "amount",
        },
    )
    assert repeated == {1}


def test_company_a_multiple_erp_candidates_use_description_and_keep_receipts():
    erp = pd.DataFrame(
        [
            {"po": "1234567890", "item": "10", "supplier": "acme", "amount": 100.0, "description": "steel bolt"},
            {"po": "1234567890", "item": "10", "supplier": "acme", "amount": 100.0, "description": "mine pump"},
        ]
    )
    tool = pd.DataFrame(
        [{"po": "1234567890 / 10", "item": "10", "supplier": "acme", "amount": 100.0, "description": "steel bolt"}]
    )
    fields = {
        "erp_po": "po",
        "tool_po": "po",
        "erp_item": "item",
        "tool_item": "item",
        "erp_supplier": "supplier",
        "tool_supplier": "supplier",
        "erp_amount": "amount",
        "tool_amount": "amount",
        "tool_amount_is_net": True,
        "erp_description": ["description"],
        "tool_description": "description",
    }

    matches, ambiguous = match_a_tool_to_erp_diagnostics(erp, tool, fields)

    assert matches == {0: 0}
    assert ambiguous == set()
    assert len(erp) == 2  # Separate ERP receipts are never removed by matching.


def test_company_a_ambiguous_multiple_erp_candidates_are_not_matched():
    erp = pd.DataFrame(
        [
            {"po": "1234567890", "item": "10", "supplier": "acme", "amount": 100.0, "description": "same item"},
            {"po": "1234567890", "item": "10", "supplier": "acme", "amount": 100.0, "description": "same item"},
        ]
    )
    tool = pd.DataFrame(
        [{"po": "1234567890", "item": "10", "supplier": "acme", "amount": 100.0, "description": "same item"}]
    )
    fields = {
        "erp_po": "po", "tool_po": "po", "erp_item": "item", "tool_item": "item",
        "erp_supplier": "supplier", "tool_supplier": "supplier",
        "erp_amount": "amount", "tool_amount": "amount", "tool_amount_is_net": True,
        "erp_description": ["description"], "tool_description": "description",
    }

    matches, ambiguous = match_a_tool_to_erp_diagnostics(erp, tool, fields)

    assert matches == {}
    assert ambiguous == {0}


def test_company_a_single_amount_match_is_not_enough_among_multiple_receipts():
    erp = pd.DataFrame(
        [
            {"po": "1234567890", "item": "10", "supplier": "acme", "amount": 100.0, "description": "same item"},
            {"po": "1234567890", "item": "10", "supplier": "acme", "amount": 80.0, "description": "same item"},
        ]
    )
    tool = pd.DataFrame(
        [{"po": "1234567890", "item": "10", "supplier": "acme", "amount": 100.0, "description": "same item"}]
    )
    fields = {
        "erp_po": "po", "tool_po": "po", "erp_item": "item", "tool_item": "item",
        "erp_supplier": "supplier", "tool_supplier": "supplier",
        "erp_amount": "amount", "tool_amount": "amount", "tool_amount_is_net": True,
        "erp_description": ["description"], "tool_description": "description",
    }

    matches, ambiguous = match_a_tool_to_erp_diagnostics(erp, tool, fields)

    assert matches == {}
    assert ambiguous == {0}


def test_company_b_usd_passes_through_even_with_non_one_rate():
    result = _company_b_inputs(
        [
            {"line_id": "U1", "currency": "USD", "line_amount": 123.45, "currency_rate": 1, "invoice_date": "2024-01-15"},
            {"line_id": "U2", "currency": "USD", "line_amount": 234.56, "currency_rate": 17.5, "invoice_date": "2024-01-15"},
        ],
        [],
    ).set_index("line_id")

    assert result.loc["U1", "spend_usd"] == pytest.approx(123.45)
    assert result.loc["U2", "spend_usd"] == pytest.approx(234.56)
    assert result.loc["U2", "fx_source"] == "USD_PASSTHROUGH"
    assert not result.loc[["U1", "U2"], "fx_missing"].any()


def test_company_b_non_usd_transaction_and_monthly_fx_conversion():
    result = _company_b_inputs(
        [
            {"line_id": "E1", "currency": "EUR", "line_amount": 100.0, "currency_rate": 2.0, "invoice_date": "2024-01-15"},
            {"line_id": "E2", "currency": "EUR", "line_amount": 100.0, "currency_rate": None, "invoice_date": "2024-01-15"},
        ],
        [("EUR", "2024-01", 4.0)],
    ).set_index("line_id")

    assert result.loc["E1", "spend_usd"] == pytest.approx(50.0)
    assert result.loc["E1", "fx_source"] == "TRANSACTION_RATE"
    assert result.loc["E2", "spend_usd"] == pytest.approx(25.0)
    assert result.loc["E2", "fx_source"] == "MONTHLY_FX"


def test_company_b_missing_non_usd_fx_and_amount_remain_distinguishable():
    result = _company_b_inputs(
        [
            {"line_id": "M1", "currency": "JPY", "line_amount": 100.0, "currency_rate": None, "invoice_date": "2024-01-15"},
            {"line_id": "M2", "currency": "EUR", "line_amount": None, "currency_rate": 2.0, "invoice_date": "2024-01-15"},
            {"line_id": "Z1", "currency": "EUR", "line_amount": 0.0, "currency_rate": 2.0, "invoice_date": "2024-01-15"},
        ],
        [],
    ).set_index("line_id")

    assert pd.isna(result.loc["M1", "spend_usd"])
    assert bool(result.loc["M1", "fx_missing"])
    assert not bool(result.loc["M1", "amount_missing"])
    assert pd.isna(result.loc["M2", "spend_usd"])
    assert not bool(result.loc["M2", "fx_missing"])
    assert bool(result.loc["M2", "amount_missing"])
    assert result.loc["Z1", "spend_usd"] == 0
    assert not bool(result.loc["Z1", "spend_missing"])


def test_output_builder_zeros_excluded_spend_and_rejects_duplicate_ids():
    output = build_output(
        pd.DataFrame(
            [
                {"record_id": "A1", "include": 0, "exclusion_reason": "DUPLICATE", "spend": 42.0},
                {"record_id": "A2", "include": 1, "spend": 0.0},
            ]
        )
    )
    assert output.loc[0, "spend"] == 0
    assert list(output.columns) == [
        "record_id", "include", "exclusion_reason", "supplier_key", "item_key",
        "spend", "l1", "l2", "l3", "l4", "confidence",
    ]

    with pytest.raises(ValueError, match="Duplicate record_id"):
        build_output(pd.DataFrame([{"record_id": "A1"}, {"record_id": "A1"}]))


def test_classifier_predictions_are_taxonomy_paths():
    taxonomy = pd.DataFrame(
        {
            "category_l1": ["Operations", "Materials"],
            "category_l2": ["Services", "Parts"],
            "category_l3": ["Maintenance", "Fasteners"],
        }
    )
    training = pd.DataFrame(
        {
            "text": ["pump maintenance repair", "steel bolt nut"],
            "category_l1": taxonomy.category_l1,
            "category_l2": taxonomy.category_l2,
            "category_l3": taxonomy.category_l3,
        }
    )
    classifier = TaxonomyClassifier()
    classifier.fit(training, ["text"], ["category_l1", "category_l2", "category_l3"])
    predictions = classifier.predict(
        pd.DataFrame({"text": ["pump repair", "steel bolt"]}),
        ["text"],
        taxonomy,
    )
    allowed = set(map(tuple, taxonomy.values.tolist()))
    paths = set(zip(predictions.l1, predictions.l2, predictions.l3))

    assert paths <= allowed
