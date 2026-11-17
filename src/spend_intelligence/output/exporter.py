from pathlib import Path

import pandas as pd

from .validator import REQUIRED_COLUMNS, validate_output


def export_output(frame: pd.DataFrame, path: str | Path) -> None:
    output = frame.reindex(columns=REQUIRED_COLUMNS).copy()
    validate_output(output)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, float_format="%.6f")
