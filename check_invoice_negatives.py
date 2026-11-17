
from src.spend_intelligence.config import load_config
from src.spend_intelligence.ingestion.loader import load_company_inputs
from src.spend_intelligence.preprocessing.company_a import prepare_company_a

config = load_config("configs/company_a.yaml")
inputs = load_company_inputs(config)
processed = prepare_company_a(inputs, config)

tool = processed["invoice_tool"]

print("Negative gross amounts:",
      (tool["gross_amount_numeric"] < 0).sum())

print("Negative net amounts:",
      (tool["spend_zar"] < 0).sum())

print("\nNegative amount examples:")
print(
    tool.loc[
        tool["spend_zar"] < 0,
        ["gross_amount_numeric", "vat_rate_used", "spend_zar"]
    ].head(15)
)
