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

print("=" * 75)
print("NAME + ADDRESS UNION BLOCKING EVALUATION")
print("=" * 75)

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
# BUILD S1 LOOKUP
# ============================================================

print("\nBuilding S1 lookup...")

s1_lookup = {}

for row in s1[
    [
        "entity_id",
        "block_key",
        "country_key",
        "address_key",
    ]
].itertuples(index=False):

    s1_lookup[row.entity_id] = {
        "name_block": str(row.block_key),
        "address_block": (
            str(row.country_key)
            + "|"
            + str(row.address_key)
        ),
    }

print(
    f"S1 IDs in lookup: "
    f"{len(s1_lookup):,}"
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
# BUILD TRUE PAIR LISTS
# ============================================================

print("\nExpanding true pairs...")

true_s2 = []
true_s3 = []

for row in gt.itertuples(index=False):

    s1_id = row.source1_entity_id
    matched = row.matched_entity_ids

    if pd.isna(matched):
        continue

    matched = str(matched)

    for target_id in matched.split(","):

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
# LOAD TARGET SOURCE
# ============================================================

def load_target_lookup(
    source_path,
    source_name,
):

    print("\n" + "=" * 75)
    print(f"LOADING {source_name}")
    print("=" * 75)

    target_lookup = {}

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

        for row in chunk[
            [
                "entity_id",
                "block_key",
                "country_key",
                "address_key",
            ]
        ].itertuples(index=False):

            target_lookup[row.entity_id] = {
                "name_block": str(row.block_key),
                "address_block": (
                    str(row.country_key)
                    + "|"
                    + str(row.address_key)
                ),
            }

    print(
        f"{source_name} IDs loaded: "
        f"{len(target_lookup):,}"
    )

    return target_lookup


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    true_pairs,
    target_lookup,
    source_name,
):

    print("\n" + "=" * 75)
    print(f"EVALUATING {source_name}")
    print("=" * 75)

    total = len(true_pairs)

    name_only = 0
    address_only = 0
    both = 0
    neither = 0
    missing_target = 0

    # --------------------------------------------------------
    # IMPORTANT DEBUG COUNTERS
    # --------------------------------------------------------

    name_recovered = 0
    address_recovered = 0
    union_recovered = 0

    # --------------------------------------------------------
    # Test every known true pair
    # --------------------------------------------------------

    for s1_id, target_id in true_pairs:

        s1_record = s1_lookup.get(s1_id)
        target_record = target_lookup.get(target_id)

        if s1_record is None:
            print(
                f"WARNING: S1 missing: {s1_id}"
            )
            continue

        if target_record is None:

            missing_target += 1

            continue

        name_match = (
            s1_record["name_block"]
            ==
            target_record["name_block"]
        )

        address_match = (
            s1_record["address_block"]
            ==
            target_record["address_block"]
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        if name_match:
            name_recovered += 1

        if address_match:
            address_recovered += 1

        if name_match and address_match:

            both += 1

        elif name_match:

            name_only += 1

        elif address_match:

            address_only += 1

        else:

            neither += 1

        if name_match or address_match:
            union_recovered += 1


    # ========================================================
    # METRICS
    # ========================================================

    valid_total = total - missing_target

    name_recall = (
        name_recovered / valid_total
        if valid_total
        else 0
    )

    address_recall = (
        address_recovered / valid_total
        if valid_total
        else 0
    )

    union_recall = (
        union_recovered / valid_total
        if valid_total
        else 0
    )


    # ========================================================
    # RESULTS
    # ========================================================

    print("\nResults:")
    print("-" * 75)

    print(
        f"Total true pairs:       "
        f"{total:,}"
    )

    print(
        f"Missing target:         "
        f"{missing_target:,}"
    )

    print()

    print(
        f"Name recovered:         "
        f"{name_recovered:,}"
    )

    print(
        f"Address recovered:      "
        f"{address_recovered:,}"
    )

    print(
        f"Union recovered:        "
        f"{union_recovered:,}"
    )

    print()

    print(
        f"Name recall:            "
        f"{name_recall:.4%}"
    )

    print(
        f"Address recall:         "
        f"{address_recall:.4%}"
    )

    print(
        f"Union recall:           "
        f"{union_recall:.4%}"
    )

    print()

    print(
        f"Both name + address:    "
        f"{both:,}"
    )

    print(
        f"Name only:              "
        f"{name_only:,}"
    )

    print(
        f"Address only:           "
        f"{address_only:,}"
    )

    print(
        f"Neither:                "
        f"{neither:,}"
    )

    print()

    print(
        f"Check: both + name_only "
        f"+ address_only + neither"
    )

    print(
        both
        + name_only
        + address_only
        + neither
    )

    return {
        "total": total,
        "name_recovered": name_recovered,
        "address_recovered": address_recovered,
        "union_recovered": union_recovered,
        "both": both,
        "name_only": name_only,
        "address_only": address_only,
        "neither": neither,
        "missing_target": missing_target,
        "name_recall": name_recall,
        "address_recall": address_recall,
        "union_recall": union_recall,
    }


# ============================================================
# S2
# ============================================================

s2_lookup = load_target_lookup(
    S2_PATH,
    "S2",
)

s2_results = evaluate(
    true_s2,
    s2_lookup,
    "S2",
)


# ============================================================
# FREE MEMORY
# ============================================================

del s2_lookup


# ============================================================
# S3
# ============================================================

s3_lookup = load_target_lookup(
    S3_PATH,
    "S3",
)

s3_results = evaluate(
    true_s3,
    s3_lookup,
    "S3",
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("NAME + ADDRESS UNION BLOCKING SUMMARY")
print("=" * 75)


print("\nS2")
print("-" * 75)

print(
    f"Name recall:       "
    f"{s2_results['name_recall']:.4%}"
)

print(
    f"Address recall:    "
    f"{s2_results['address_recall']:.4%}"
)

print(
    f"Union recall:      "
    f"{s2_results['union_recall']:.4%}"
)

print(
    f"Address-only:      "
    f"{s2_results['address_only']:,}"
)


print("\nS3")
print("-" * 75)

print(
    f"Name recall:       "
    f"{s3_results['name_recall']:.4%}"
)

print(
    f"Address recall:    "
    f"{s3_results['address_recall']:.4%}"
)

print(
    f"Union recall:      "
    f"{s3_results['union_recall']:.4%}"
)

print(
    f"Address-only:      "
    f"{s3_results['address_only']:,}"
)


print("\n" + "=" * 75)
print("EVALUATION COMPLETE")
print("=" * 75)