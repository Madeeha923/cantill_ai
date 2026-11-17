from src.spend_intelligence.config import load_config
from src.spend_intelligence.ingestion.loader import load_company_inputs

for company in ["a", "b"]:
    config = load_config(f"configs/company_{company}.yaml")
    data = load_company_inputs(config)

    labels = data["labels"]

    print(f"\nCOMPANY {company.upper()} LABEL COLUMNS")
    print(labels.columns.tolist())
    print(labels.head(3).to_string(index=False))

    taxonomy = data["taxonomy"]

    print(f"\nCOMPANY {company.upper()} TAXONOMY COLUMNS")
    print(taxonomy.columns.tolist())
    print(taxonomy.head(3).to_string(index=False))