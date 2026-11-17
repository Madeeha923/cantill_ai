import pandas as pd


def convert_to_reporting_currency(amount, currency, rate=None, fx_rates=None, date=None, reporting_currency="USD"):
    """Convert amounts using units-of-source-currency per one reporting-currency unit."""
    amount = pd.to_numeric(amount, errors="coerce")
    currency = str(currency or "").strip().upper()
    reporting_currency = reporting_currency.upper()
    if pd.isna(amount):
        return 0.0, "INVALID_AMOUNT"
    if currency == reporting_currency:
        return float(amount), "OK"
    numeric_rate = pd.to_numeric(rate, errors="coerce")
    if pd.notna(numeric_rate) and numeric_rate > 0:
        return float(amount / numeric_rate), "SOURCE_RATE"
    if fx_rates is not None and date is not None:
        month = pd.to_datetime(date, errors="coerce")
        if pd.notna(month):
            month = month.strftime("%Y-%m")
            match = fx_rates[(fx_rates["currency"].astype(str).str.upper() == currency) & (fx_rates["month"].astype(str) == month)]
            if not match.empty:
                fx = pd.to_numeric(match.iloc[0]["units_per_usd"], errors="coerce")
                if pd.notna(fx) and fx > 0:
                    return float(amount / fx), "MONTHLY_FX"
    return 0.0, "MISSING_FX"
