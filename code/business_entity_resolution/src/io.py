from pathlib import Path
import pandas as pd


def read_tsv(path: str | Path) -> pd.DataFrame:
    """
    Read a competition TSV file.
    """
    return pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
    )