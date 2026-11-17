import pandas as pd


def parse_dates(series: pd.Series, dayfirst: bool = False) -> tuple[pd.Series, pd.Series]:
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, format="mixed")
    invalid = series.notna() & series.astype(str).str.strip().ne("") & parsed.isna()
    return parsed, invalid
