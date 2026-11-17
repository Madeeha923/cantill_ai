from collections.abc import Iterable

import pandas as pd


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source_name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source_name} is missing configured columns: {', '.join(missing)}")
