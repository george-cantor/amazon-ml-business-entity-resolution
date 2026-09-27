from pathlib import Path
import pandas as pd

from blocking import add_block_key


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "train"

S1_PATH = DATA_DIR / "train_source1.tsv"
S2_PATH = DATA_DIR / "train_source2.tsv"
GT_PATH = DATA_DIR / "train_ground_truth.tsv"


# ============================================================
# LOAD ONE GROUND-TRUTH ROW
# ============================================================

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str,
    usecols=[
        "source1_entity_id",
        "matched_entity_ids",
    ],
)

row = gt.iloc[0]

s1_id = row["source1_entity_id"]

matched = str(row["matched_entity_ids"])

target_ids = [
    x.strip()
    for x in matched.split(",")
    if x.strip()
]

s2_id = next(
    x for x in target_ids
    if x.startswith("S2-")
)


print("=" * 70)
print("DEBUGGING ONE TRUE PAIR")
print("=" * 70)

print("\nS1 ID:")
print(s1_id)

print("\nS2 ID:")
print(s2_id)


# ============================================================
# LOAD S1 RECORD
# ============================================================

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ],
)

s1 = add_block_key(s1)

s1_row = s1[
    s1["entity_id"] == s1_id
].iloc[0]


# ============================================================
# LOAD S2 RECORD
# ============================================================

s2 = pd.read_csv(
    S2_PATH,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ],
)

s2 = add_block_key(s2)

s2_row = s2[
    s2["entity_id"] == s2_id
].iloc[0]


# ============================================================
# PRINT RECORDS
# ============================================================

print("\n" + "=" * 70)
print("S1 RECORD")
print("=" * 70)

print(
    s1_row[
        [
            "entity_id",
            "business_name",
            "country",
            "name_key",
            "block_key",
        ]
    ].to_string()
)


print("\n" + "=" * 70)
print("S2 RECORD")
print("=" * 70)

print(
    s2_row[
        [
            "entity_id",
            "business_name",
            "country",
            "name_key",
            "block_key",
        ]
    ].to_string()
)


# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

print(
    "\nS1 name_key: "
    + repr(s1_row["name_key"])
)

print(
    "S2 name_key: "
    + repr(s2_row["name_key"])
)

print(
    "\nS1 block_key: "
    + repr(s1_row["block_key"])
)

print(
    "S2 block_key: "
    + repr(s2_row["block_key"])
)

print(
    "\nName block equal:",
    s1_row["block_key"]
    == s2_row["block_key"]
)

print("\nS1 country:")
print(repr(s1_row["country"]))

print("S2 country:")
print(repr(s2_row["country"]))

print("\nS1 country_key:")
print(repr(s1_row["country_key"]))

print("S2 country_key:")
print(repr(s2_row["country_key"]))

print("\nS1 address_key:")
print(repr(s1_row["address_key"]))

print("S2 address_key:")
print(repr(s2_row["address_key"]))

print(
    "\nAddress block equal:",
    (
        s1_row["country_key"]
        + "|"
        + s1_row["address_key"]
    )
    ==
    (
        s2_row["country_key"]
        + "|"
        + s2_row["address_key"]
    )
)