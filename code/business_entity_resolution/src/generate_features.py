from pathlib import Path
import sys
import pandas as pd

# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "train"

S1_PATH = DATA_DIR / "train_source1.tsv"
S2_PATH = DATA_DIR / "train_source2.tsv"
S3_PATH = DATA_DIR / "train_source3.tsv"

TRAINING_PAIRS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "training_pairs"
    / "training_pairs.tsv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "features"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "training_features.tsv"

CHUNKSIZE = 200_000


# ------------------------------------------------------------
# IMPORT EXISTING FEATURE FUNCTIONS
# ------------------------------------------------------------

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features import build_feature_table


# ------------------------------------------------------------
# LOAD ONLY REQUIRED SOURCE RECORDS
# ------------------------------------------------------------

def load_required_records(path, required_ids, source_name):

    print(f"\nLoading required records from {source_name}...")

    required_ids = set(required_ids)

    found = {}

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            chunksize=CHUNKSIZE
        ),
        start=1
    ):

        mask = chunk["entity_id"].isin(required_ids)

        matched = chunk.loc[
            mask,
            [
                "entity_id",
                "business_name",
                "business_address",
                "country"
            ]
        ]

        if not matched.empty:

            for row in matched.itertuples(index=False):

                found[row.entity_id] = {
                    "business_name": row.business_name,
                    "business_address": row.business_address,
                    "country": row.country
                }

        print(
            f"  {source_name} chunk {chunk_no}: "
            f"found {len(found):,} records"
        )

        if len(found) == len(required_ids):
            print(f"  All required {source_name} records found.")
            break

    return pd.DataFrame(
        [
            {
                "entity_id": entity_id,
                **values
            }
            for entity_id, values in found.items()
        ]
    )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("PHASE 4 — FEATURE GENERATION")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD TRAINING PAIRS
    # --------------------------------------------------------

    print("\nLoading training pairs...")

    pair_df = pd.read_csv(
        TRAINING_PAIRS_PATH,
        sep="\t",
        dtype={
            "source1_entity_id": str,
            "target_entity_id": str,
            "target_source": str,
            "y": int
        }
    )

    print(f"Training pairs: {len(pair_df):,}")

    print("\nTarget-source distribution:")
    print(pair_df["target_source"].value_counts())

    print("\nLabel distribution:")
    print(pair_df["y"].value_counts())

    # --------------------------------------------------------
    # GET ONLY THE IDs WE ACTUALLY NEED
    # --------------------------------------------------------

    s1_ids = set(
        pair_df["source1_entity_id"].dropna()
    )

    s2_ids = set(
        pair_df.loc[
            pair_df["target_source"] == "S2",
            "target_entity_id"
        ].dropna()
    )

    s3_ids = set(
        pair_df.loc[
            pair_df["target_source"] == "S3",
            "target_entity_id"
        ].dropna()
    )

    print("\nRequired records:")
    print(f"S1 IDs: {len(s1_ids):,}")
    print(f"S2 IDs: {len(s2_ids):,}")
    print(f"S3 IDs: {len(s3_ids):,}")

    # --------------------------------------------------------
    # LOAD REQUIRED S1
    # --------------------------------------------------------

    s1_df = load_required_records(
        S1_PATH,
        s1_ids,
        "S1"
    )

    # --------------------------------------------------------
    # LOAD REQUIRED S2
    # --------------------------------------------------------

    s2_df = load_required_records(
        S2_PATH,
        s2_ids,
        "S2"
    )

    # --------------------------------------------------------
    # LOAD REQUIRED S3
    # --------------------------------------------------------

    s3_df = load_required_records(
        S3_PATH,
        s3_ids,
        "S3"
    )

    print("\nRecords loaded:")
    print(f"S1: {len(s1_df):,}")
    print(f"S2: {len(s2_df):,}")
    print(f"S3: {len(s3_df):,}")

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    missing_s1 = s1_ids - set(s1_df["entity_id"])
    missing_s2 = s2_ids - set(s2_df["entity_id"])
    missing_s3 = s3_ids - set(s3_df["entity_id"])

    print("\nMissing records:")
    print(f"S1 missing: {len(missing_s1):,}")
    print(f"S2 missing: {len(missing_s2):,}")
    print(f"S3 missing: {len(missing_s3):,}")

    if missing_s1 or missing_s2 or missing_s3:

        print(
            "\nWARNING: Some records required for feature "
            "generation were not found."
        )

    # --------------------------------------------------------
    # BUILD FEATURES
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BUILDING FEATURES")
    print("=" * 70)

    print(
        "\nCalculating pairwise similarity features.\n"
        "This may take several minutes because the feature "
        "functions calculate fuzzy string similarities."
    )

    feature_df = build_feature_table(
        pair_df,
        s1_df,
        s2_df,
        s3_df
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE GENERATION RESULTS")
    print("=" * 70)

    print(f"\nInput training pairs:  {len(pair_df):,}")
    print(f"Feature rows generated: {len(feature_df):,}")

    print("\nFeature columns:")

    for column in feature_df.columns:
        print(f"  {column}")

    print("\nLabel distribution:")

    if "y" in feature_df.columns:
        print(feature_df["y"].value_counts())

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    feature_df.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False
    )

    print("\nSaved to:")
    print(OUTPUT_PATH)

    print("\nFeature generation complete.")


if __name__ == "__main__":
    main()