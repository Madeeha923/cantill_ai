
from src.spend_intelligence.config import load_config
from src.spend_intelligence.ingestion.loader import load_company_inputs

config = load_config("configs/company_a.yaml")

datasets = load_company_inputs(config)

for name, df in datasets.items():
    print(name, df.shape)