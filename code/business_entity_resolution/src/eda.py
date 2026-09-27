import pandas as pd
from pathlib import Path


# ============================================================
# PHASE 1 — TRAINING DATA EDA
# ============================================================

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Training Source 1 path
SOURCE1_PATH = PROJECT_ROOT / "data" / "train" / "train_source1.tsv"


print("=" * 70)
print("PHASE 1 — SOURCE 1 DATASET EDA")
print("=" * 70)

print("\nLoading Source 1...")
print(f"File: {SOURCE1_PATH}")


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df1 = pd.read_csv(
    SOURCE1_PATH,
    sep="\t",
    dtype="string",
    keep_default_na=False
)


# ------------------------------------------------------------
# Basic information
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("1. BASIC INFORMATION")
print("=" * 70)

print(f"Rows: {len(df1):,}")
print(f"Columns: {len(df1.columns)}")

print("\nColumns:")
for col in df1.columns:
    print(f"  - {col}")


# ------------------------------------------------------------
# First few rows
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("2. FIRST 5 ROWS")
print("=" * 70)

print(df1.head().to_string(index=False))


# ------------------------------------------------------------
# Data types
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3. DATA TYPES")
print("=" * 70)

print(df1.dtypes)


# ------------------------------------------------------------
# Empty values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("4. EMPTY VALUES")
print("=" * 70)

for col in df1.columns:
    empty_count = (df1[col].str.strip() == "").sum()

    print(
        f"{col:<20} : "
        f"{empty_count:,} empty values "
        f"({empty_count / len(df1) * 100:.2f}%)"
    )


# ------------------------------------------------------------
# Duplicate entity IDs
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("5. ENTITY ID CHECK")
print("=" * 70)

total_rows = len(df1)
unique_ids = df1["entity_id"].nunique()
duplicate_ids = df1["entity_id"].duplicated().sum()

print(f"Total rows       : {total_rows:,}")
print(f"Unique entity IDs: {unique_ids:,}")
print(f"Duplicate IDs    : {duplicate_ids:,}")


# ------------------------------------------------------------
# Country distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("6. COUNTRY DISTRIBUTION")
print("=" * 70)

print(
    df1["country"]
    .value_counts(dropna=False)
    .to_string()
)


# ------------------------------------------------------------
# Business name length
# ------------------------------------------------------------

df1["name_length"] = df1["business_name"].str.len()

print("\n" + "=" * 70)
print("7. BUSINESS NAME LENGTH")
print("=" * 70)

print(df1["name_length"].describe())


# ------------------------------------------------------------
# Address length
# ------------------------------------------------------------

df1["address_length"] = df1["business_address"].str.len()

print("\n" + "=" * 70)
print("8. BUSINESS ADDRESS LENGTH")
print("=" * 70)

print(df1["address_length"].describe())


# ------------------------------------------------------------
# Very short names
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("9. VERY SHORT BUSINESS NAMES")
print("=" * 70)

short_names = (
    df1[
        df1["business_name"].str.strip().str.len() <= 3
    ][
        ["entity_id", "business_name", "business_address", "country"]
    ]
    .head(20)
)

print(short_names.to_string(index=False))


# ------------------------------------------------------------
# Very long names
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("10. LONG BUSINESS NAMES")
print("=" * 70)

long_names = (
    df1.sort_values("name_length", ascending=False)
    [
        ["entity_id", "business_name", "business_address", "country"]
    ]
    .head(20)
)

print(long_names.to_string(index=False))


# ------------------------------------------------------------
# Sample records
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("11. RANDOM SAMPLE")
print("=" * 70)

print(
    df1[
        ["entity_id", "business_name", "business_address", "country"]
    ]
    .sample(min(20, len(df1)), random_state=42)
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("SOURCE 1 EDA COMPLETE")
print("=" * 70)

# ============================================================
# GROUND TRUTH ANALYSIS
# ============================================================

GT_PATH = PROJECT_ROOT / "data" / "train" / "train_ground_truth.tsv"

print("\n" + "=" * 70)
print("PHASE 1 — GROUND TRUTH ANALYSIS")
print("=" * 70)

print(f"\nLoading: {GT_PATH}")

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype="string",
    keep_default_na=False
)

print("\nShape:")
print(gt.shape)

print("\nColumns:")
print(gt.columns.tolist())

print("\nFirst 10 rows:")
print(gt.head(10).to_string(index=False))

# ------------------------------------------------------------
# Number of matches per Source-1 entity
# ------------------------------------------------------------

def count_matches(x):
    if not x or str(x).strip() == "":
        return 0

    return len([
        item.strip()
        for item in str(x).split(",")
        if item.strip()
    ])


gt["n_matches"] = gt["matched_entity_ids"].apply(count_matches)

print("\n" + "=" * 70)
print("MATCH COUNT DISTRIBUTION")
print("=" * 70)

print(
    gt["n_matches"]
    .value_counts()
    .sort_index()
    .to_string()
)

# ------------------------------------------------------------
# Match statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MATCH STATISTICS")
print("=" * 70)

print(gt["n_matches"].describe())

print(
    f"\nNo-match entities: "
    f"{(gt['n_matches'] == 0).sum():,}"
)

print(
    f"No-match percentage: "
    f"{(gt['n_matches'] == 0).mean() * 100:.2f}%"
)

print(
    f"Entities with >=1 match: "
    f"{(gt['n_matches'] > 0).sum():,}"
)

print(
    f"Entities with multiple matches: "
    f"{(gt['n_matches'] > 1).sum():,}"
)

# ------------------------------------------------------------
# S2 / S3 match distribution
# ------------------------------------------------------------

def source_counts(x):

    if not x:
        return 0, 0

    ids = [
        item.strip()
        for item in str(x).split(",")
        if item.strip()
    ]

    n_s2 = sum(entity_id.startswith("S2-") for entity_id in ids)
    n_s3 = sum(entity_id.startswith("S3-") for entity_id in ids)

    return n_s2, n_s3


gt[["n_s2", "n_s3"]] = gt["matched_entity_ids"].apply(
    lambda x: pd.Series(source_counts(x))
)

print("\n" + "=" * 70)
print("S2 / S3 MATCH COUNTS")
print("=" * 70)

print(f"Total S2 matches: {gt['n_s2'].sum():,}")
print(f"Total S3 matches: {gt['n_s3'].sum():,}")

print(
    f"S1 entities with S2 matches: "
    f"{(gt['n_s2'] > 0).sum():,}"
)

print(
    f"S1 entities with S3 matches: "
    f"{(gt['n_s3'] > 0).sum():,}"
)
