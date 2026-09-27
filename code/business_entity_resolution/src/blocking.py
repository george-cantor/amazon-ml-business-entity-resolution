from pathlib import Path
import re
import unicodedata

import pandas as pd


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "train"

S1_PATH = DATA_DIR / "train_source1.tsv"
S2_PATH = DATA_DIR / "train_source2.tsv"
S3_PATH = DATA_DIR / "train_source3.tsv"


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize business names / addresses using the same
    Unicode-aware logic used during Phase 1.
    """

    if text is None:
        return ""

    text = str(text)

    text = unicodedata.normalize("NFKC", text)

    text = text.casefold()

    text = text.replace("&", " and ")

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# ============================================================
# NAME BLOCKING KEY
# ============================================================

def name_key(name: str) -> str:
    """
    Create the Phase 1 blocking key.

    The key consists of the first four Unicode-aware
    alphanumeric characters after normalization.
    """

    normalized = normalize_text(name)

    if not normalized:
        return ""

    compact = "".join(
        char
        for char in normalized
        if char.isalnum()
    )

    return compact[:4]


# ============================================================
# VECTORIZED NAME BLOCKING KEY
# ============================================================

def name_key_series(series: pd.Series) -> pd.Series:
    """
    Vectorized version of name_key() for pandas Series.
    """

    s = (
        series
        .fillna("")
        .astype("string")
        .str.normalize("NFKC")
        .str.casefold()
        .str.replace("&", " and ", regex=False)
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return (
        s
        .str.replace(r"[^\w]", "", regex=True)
        .str.slice(0, 4)
        .fillna("")
    )

def address_key_series(series: pd.Series) -> pd.Series:
    """
    Vectorized address blocking key.

    Uses the normalized address and retains
    the first 8 characters for experimental blocking.
    """

    s = (
        series
        .fillna("")
        .astype("string")
        .str.normalize("NFKC")
        .str.casefold()
        .str.replace("&", " and ", regex=False)
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    compact = s.str.replace(
        r"\s+",
        "",
        regex=True,
    )

    return compact.str.slice(0, 8).fillna("")

# ============================================================
# BLOCK KEY
# ============================================================

def add_block_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the first blocking key:

        country + first four normalized name characters
    """

    result = df.copy()

    result["country_key"] = (
        result["country"]
        .fillna("")
        .astype("string")
        .str.casefold()
        .str.strip()
    )

    result["name_key"] = name_key_series(
        result["business_name"]
    )
    result["address_key"] = address_key_series(
        result["business_address"]
    )

    result["block_key"] = (
        result["country_key"]
        + "|"
        + result["name_key"]
    )

    return result