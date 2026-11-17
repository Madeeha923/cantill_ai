from pathlib import Path

import pandas as pd


def load_taxonomy(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    tree = pd.read_csv(path, dtype="string").fillna("")
    columns = columns or [c for c in tree.columns if c.lower().startswith("category_l") or c.lower().startswith("l")]
    if not columns:
        raise ValueError(f"No taxonomy level columns found in {path}")
    for column in columns:
        if column not in tree.columns:
            tree[column] = ""
    tree = tree[columns].apply(lambda col: col.astype(str).str.strip())
    tree = tree[tree[columns[0]].ne("")].drop_duplicates().reset_index(drop=True)
    return tree


def taxonomy_paths(tree: pd.DataFrame) -> list[tuple[str, ...]]:
    return [tuple(value for value in row if value) for row in tree.itertuples(index=False, name=None)]
