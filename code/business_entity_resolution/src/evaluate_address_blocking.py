from pathlib import Path
import pandas as pd

from blocking import add_block_key


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "train"

S1_PATH = DATA_DIR / "train_source1.tsv"
S2_PATH = DATA_DIR / "train_source2.tsv"
S3_PATH = DATA_DIR / "train_source3.tsv"
GT_PATH = DATA_DIR / "train_ground_truth.tsv"

CHUNKSIZE = 200_000


# ============================================================
# LOAD S1
# ============================================================

print("=" * 70)
print("ADDRESS BLOCKING RECALL EVALUATION")
print("=" * 70)

print("\nLoading S1...")

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

print(f"S1 rows: {len(s1):,}")


# ============================================================
# S1 ADDRESS BLOCK LOOKUP
# ============================================================

print("\nBuilding S1 address-block lookup...")

s1_address_lookup = dict(
    zip(
        s1["entity_id"],
        s1["country_key"] + "|" + s1["address_key"],
    )
)

print(
    f"S1 IDs in lookup: "
    f"{len(s1_address_lookup):,}"
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str,
    usecols=[
        "source1_entity_id",
        "matched_entity_ids",
    ],
)

print(
    f"Ground-truth rows: "
    f"{len(gt):,}"
)


# ============================================================
# EXPAND TRUE PAIRS
# ============================================================

print("\nExpanding true pairs...")

true_s2 = []
true_s3 = []

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]
    matched = row["matched_entity_ids"]

    if pd.isna(matched) or not str(matched).strip():
        continue

    for target_id in str(matched).split(","):

        target_id = target_id.strip()

        if not target_id:
            continue

        if target_id.startswith("S2-"):
            true_s2.append(
                (s1_id, target_id)
            )

        elif target_id.startswith("S3-"):
            true_s3.append(
                (s1_id, target_id)
            )


print(
    f"S2 true pairs: "
    f"{len(true_s2):,}"
)

print(
    f"S3 true pairs: "
    f"{len(true_s3):,}"
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_source(
    source_path,
    source_name,
    true_pairs,
):

    print("\n" + "=" * 70)
    print(f"EVALUATING {source_name}")
    print("=" * 70)

    target_address_lookup = {}

    chunk_number = 0

    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        dtype=str,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        chunksize=CHUNKSIZE,
    ):

        chunk_number += 1

        print(
            f"Reading {source_name} "
            f"chunk {chunk_number}..."
        )

        chunk = add_block_key(chunk)

        address_keys = (
            chunk["country_key"]
            + "|"
            + chunk["address_key"]
        )

        for entity_id, address_key in zip(
            chunk["entity_id"],
            address_keys,
        ):
            target_address_lookup[
                entity_id
            ] = address_key

    print(
        f"{source_name} IDs loaded: "
        f"{len(target_address_lookup):,}"
    )

    # --------------------------------------------------------
    # Evaluate true pairs
    # --------------------------------------------------------

    recovered = 0
    different_block = 0
    missing_target = 0

    for s1_id, target_id in true_pairs:

        s1_key = s1_address_lookup.get(s1_id)

        target_key = target_address_lookup.get(
            target_id
        )

        if target_key is None:
            missing_target += 1
            continue

        if s1_key == target_key:
            recovered += 1
        else:
            different_block += 1

    total_true = len(true_pairs)

    recall = (
        recovered / total_true
        if total_true
        else 0.0
    )

    print("\nResults:")
    print(
        f"Total true pairs:      "
        f"{total_true:,}"
    )

    print(
        f"Recovered by blocking: "
        f"{recovered:,}"
    )

    print(
        f"Different block:       "
        f"{different_block:,}"
    )

    print(
        f"Missing target:        "
        f"{missing_target:,}"
    )

    print(
        f"Address blocking recall: "
        f"{recall:.4%}"
    )

    return recall


# ============================================================
# S2
# ============================================================

s2_recall = evaluate_source(
    S2_PATH,
    "S2",
    true_s2,
)


# ============================================================
# S3
# ============================================================

s3_recall = evaluate_source(
    S3_PATH,
    "S3",
    true_s3,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("ADDRESS BLOCKING RECALL SUMMARY")
print("=" * 70)

print(
    f"S2 recall: {s2_recall:.4%}"
)

print(
    f"S3 recall: {s3_recall:.4%}"
)

print(
    "\nAddress blocking evaluation complete."
)