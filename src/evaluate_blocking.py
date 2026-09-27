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
print("BLOCKING RECALL EVALUATION")
print("=" * 70)

print("\nLoading S1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "country",
    ],
)

s1 = add_block_key(s1)

print(f"S1 rows: {len(s1):,}")
print(f"S1 unique blocks: {s1['block_key'].nunique():,}")


# ============================================================
# BUILD S1 BLOCK LOOKUP
# ============================================================

print("\nBuilding S1 block lookup...")

s1_block_lookup = dict(
    zip(
        s1["entity_id"],
        s1["block_key"],
    )
)

print(f"S1 IDs in lookup: {len(s1_block_lookup):,}")


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

print(f"Ground-truth rows: {len(gt):,}")


# ============================================================
# EXPAND GROUND TRUTH
# ============================================================

print("\nExpanding true pairs...")

true_pairs = []

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]
    matched = row["matched_entity_ids"]

    if pd.isna(matched) or not str(matched).strip():
        continue

    for target_id in str(matched).split(","):

        target_id = target_id.strip()

        if target_id:
            true_pairs.append(
                (
                    s1_id,
                    target_id,
                )
            )

print(f"Known true pairs: {len(true_pairs):,}")


# ============================================================
# SPLIT TRUE PAIRS BY SOURCE
# ============================================================

true_s2 = [
    pair
    for pair in true_pairs
    if pair[1].startswith("S2-")
]

true_s3 = [
    pair
    for pair in true_pairs
    if pair[1].startswith("S3-")
]

print(f"S2 true pairs: {len(true_s2):,}")
print(f"S3 true pairs: {len(true_s3):,}")


# ============================================================
# EVALUATE ONE SOURCE
# ============================================================

def evaluate_source(source_path, source_name, true_pairs):

    print("\n" + "=" * 70)
    print(f"EVALUATING {source_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Build lookup of target entity -> block key
    # --------------------------------------------------------

    target_block_lookup = {}

    chunk_number = 0

    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        dtype=str,
        usecols=[
            "entity_id",
            "business_name",
            "country",
        ],
        chunksize=CHUNKSIZE,
    ):

        chunk_number += 1

        print(
            f"Reading {source_name} chunk "
            f"{chunk_number}..."
        )

        chunk = add_block_key(chunk)

        for entity_id, block_key in zip(
            chunk["entity_id"],
            chunk["block_key"],
        ):
            target_block_lookup[
                entity_id
            ] = block_key

    print(
        f"{source_name} IDs loaded: "
        f"{len(target_block_lookup):,}"
    )

    # --------------------------------------------------------
    # Check true pairs
    # --------------------------------------------------------

    total_true = len(true_pairs)
    recovered = 0
    missing_target = 0
    different_block = 0

    for s1_id, target_id in true_pairs:

        s1_block = s1_block_lookup.get(s1_id)
        target_block = target_block_lookup.get(target_id)

        if target_block is None:
            missing_target += 1
            continue

        if s1_block == target_block:
            recovered += 1
        else:
            different_block += 1

    recall = (
        recovered / total_true
        if total_true > 0
        else 0.0
    )

    print("\nResults:")
    print(f"Total true pairs:      {total_true:,}")
    print(f"Recovered by blocking: {recovered:,}")
    print(f"Different block:       {different_block:,}")
    print(f"Missing target:        {missing_target:,}")
    print(f"Blocking recall:       {recall:.4%}")

    return {
        "source": source_name,
        "total_true": total_true,
        "recovered": recovered,
        "different_block": different_block,
        "missing_target": missing_target,
        "recall": recall,
    }


# ============================================================
# S2
# ============================================================

s2_result = evaluate_source(
    S2_PATH,
    "S2",
    true_s2,
)


# ============================================================
# S3
# ============================================================

s3_result = evaluate_source(
    S3_PATH,
    "S3",
    true_s3,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("BLOCKING RECALL SUMMARY")
print("=" * 70)

print(
    f"S2 recall: {s2_result['recall']:.4%}"
)

print(
    f"S3 recall: {s3_result['recall']:.4%}"
)

print("\nBlocking recall evaluation complete.")