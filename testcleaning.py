
import pandas as pd

from src.spend_intelligence.preprocessing.cleaning import (
    clean_text,
    parse_numeric,
    parse_dates
)

# Test 1: Text cleaning
text_data = pd.Series([
    "  Supplier ABC  ",
    "",
    "NULL",
    "Material XYZ"
])

print("TEXT CLEANING:")
print(clean_text(text_data))

# Test 2: Numeric cleaning
amounts = pd.Series([
    "5,266.63",
    "1000",
    "(500.50)",
    "invalid",
    None
])

numeric, invalid = parse_numeric(amounts)

print("\nNUMERIC CLEANING:")
print(numeric)

print("\nINVALID VALUES:")
print(invalid)

# Test 3: Date cleaning
df = pd.DataFrame({
    "invoice_date": [
        "2024-01-15",
        "invalid-date",
        "2024-03-20"
    ]
})

df = parse_dates(df, ["invoice_date"])

print("\nDATE CLEANING:")
print(df)