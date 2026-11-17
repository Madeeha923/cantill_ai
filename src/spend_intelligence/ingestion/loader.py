from pathlib import Path
import pandas as pd


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV file safely.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        low_memory=False
    )

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    return df


def load_sources(
    config: dict,
    project_root: str | Path = ".",
) -> dict[str, pd.DataFrame]:
    """Load the configured CSV sources using paths relative to project_root."""
    project_root = Path(project_root)
    sources = config.get("files")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Configuration 'files' must be a non-empty mapping of source names to paths.")
    datasets = {}
    for dataset_name, relative_path in sources.items():
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError(f"Configured path for source '{dataset_name}' must be a non-empty string.")
        file_path = project_root / relative_path
        print(f"Loading {dataset_name}: {file_path}")
        datasets[dataset_name] = load_csv(file_path)
        print(
            f"Loaded {len(datasets[dataset_name]):,} rows "
            f"and {len(datasets[dataset_name].columns)} columns"
        )
    return datasets


def load_company_inputs(
    config: dict,
    project_root: str | Path = ".",
) -> dict[str, pd.DataFrame]:
    """Backward-compatible Company A/B name for the shared source loader."""
    return load_sources(config, project_root)
