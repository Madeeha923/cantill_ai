from pathlib import Path

import pandas as pd
import pytest

from src.spend_intelligence.config import load_config
from src.spend_intelligence.output.builder import OUTPUT_COLUMNS
from src.spend_intelligence.pipeline import run_generic_pipeline


ROOT = Path(__file__).resolve().parents[1]
THREE_LEVEL_CONFIG = ROOT / "tests/fixtures/company_c_three_level/config.yaml"
FOUR_LEVEL_CONFIG = ROOT / "tests/fixtures/company_c_four_level/config.yaml"


def test_generic_pipeline_three_level_mapping_and_missing_optional_ids():
    config = load_config(THREE_LEVEL_CONFIG)
    output = run_generic_pipeline(config, project_root=ROOT)

    assert len(output) == 3
    assert list(output.columns) == OUTPUT_COLUMNS
    assert output["record_id"].tolist() == ["S001", "S002", "S003"]
    assert output["record_id"].is_unique
    assert output["supplier_key"].eq("supplier-unknown").all()
    assert output["item_key"].eq("item-unknown").all()
    assert output["confidence"].between(0, 1).all()

    taxonomy = pd.read_csv(ROOT / "tests/fixtures/company_c_three_level/taxonomy.csv").fillna("")
    valid_paths = set(
        tuple(row) for row in taxonomy[["level_one", "level_two", "level_three"]].astype(str).values
    )
    output_paths = set(zip(output["l1"], output["l2"], output["l3"]))
    assert output_paths <= valid_paths

    # A category with missing lower levels remains a valid shorter path.
    miscellaneous = output.set_index("record_id").loc["S003"]
    assert miscellaneous["l1"] == "Miscellaneous"
    assert miscellaneous["l2"] == ""
    assert miscellaneous["l3"] == ""
    assert miscellaneous["l4"] == ""


def test_generic_pipeline_four_level_taxonomy_and_configurable_columns():
    config = load_config(FOUR_LEVEL_CONFIG)
    output = run_generic_pipeline(config, project_root=ROOT)

    assert len(output) == 2
    assert list(output.columns) == OUTPUT_COLUMNS
    assert output["record_id"].tolist() == ["F001", "F002"]
    assert output["record_id"].is_unique
    assert output["confidence"].between(0, 1).all()
    assert output.set_index("record_id").loc["F001", "l4"] == "Accessories"
    assert output.set_index("record_id").loc["F002", "supplier_key"] == "supplier-unknown"

    taxonomy = pd.read_csv(ROOT / "tests/fixtures/company_c_four_level/taxonomy.csv").fillna("")
    valid_paths = set(
        tuple(row) for row in taxonomy[["tier_1", "tier_2", "tier_3", "tier_4"]].astype(str).values
    )
    output_paths = set(zip(output["l1"], output["l2"], output["l3"], output["l4"]))
    assert output_paths <= valid_paths


def test_generic_pipeline_reports_configured_column_mismatch_clearly():
    config = load_config(THREE_LEVEL_CONFIG)
    config["generic"]["record_id_column"] = "missing_record_key"

    with pytest.raises(ValueError, match="transactions source is missing configured columns: missing_record_key"):
        run_generic_pipeline(config, project_root=ROOT)


def test_company_c_config_is_schema_neutral_until_blind_data_arrives():
    config = load_config(ROOT / "configs/company_c.yaml")

    assert config["template_only"] is True
    assert config["generic"]["date_column"] is None
    assert config["generic"]["supplier_id_column"] is None
    assert config["generic"]["item_id_column"] is None
    assert config["generic"]["taxonomy_columns"] == []
