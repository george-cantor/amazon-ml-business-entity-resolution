"""
Generate supervised training pairs for the Amazon ML entity-resolution project.

Inputs:
    data/train/train_source1.tsv
    data/train/train_source2.tsv
    data/train/train_source3.tsv
    data/train/train_ground_truth.tsv
    experiments/candidate_ranking_fast/s2_top_50.tsv
    experiments/candidate_ranking_fast/s3_top_50.tsv

Output:
    experiments/training_pairs/training_pairs.tsv

The script:
1. Uses the latest Top-50 candidate files as the negative-pair pool.
2. Adds every ground-truth positive pair for the sampled S1 entities.
3. Removes duplicate pairs.
4. Labels pairs as y=1 (true match) or y=0 (candidate non-match).
5. Keeps source information (S2/S3).
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "train"
CANDIDATE_DIR = PROJECT_ROOT / "experiments" / "candidate_ranking_fast"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "training_pairs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

S1_PATH = DATA_DIR / "train_source1.tsv"
GT_PATH = DATA_DIR / "train_ground_truth.tsv"
S2_PATH = DATA_DIR / "train_source2.tsv"
S3_PATH = DATA_DIR / "train_source3.tsv"

S2_CANDIDATES = CANDIDATE_DIR / "s2_top_50.tsv"
S3_CANDIDATES = CANDIDATE_DIR / "s3_top_50.tsv"

OUTPUT_PATH = OUTPUT_DIR / "training_pairs.tsv"


def load_ground_truth():
    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    rows = []

    for row in gt.itertuples(index=False):
        s1_id = row.source1_entity_id
        matched = str(row.matched_entity_ids).strip()

        if not matched:
            continue

        for target_id in matched.split(","):
            target_id = target_id.strip()
            if target_id:
                rows.append(
                    {
                        "source1_entity_id": s1_id,
                        "target_entity_id": target_id,
                        "y": 1,
                    }
                )

    return pd.DataFrame(rows)


def load_candidates(path, source_name):
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "source1_entity_id",
        "target_entity_id",
        "target_source",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    out = df[
        ["source1_entity_id", "target_entity_id", "target_source"]
    ].copy()

    # The file itself should already contain the source, but we force it
    # here so S2/S3 remain explicit.
    out["target_source"] = source_name

    return out


def load_sampled_s1_ids():
    """
    Candidate files are generated from the sampled S1 population.
    Reading their S1 IDs is safer than assuming the experiment's sample
    size from configuration.
    """
    ids = set()

    for path in [S2_CANDIDATES, S3_CANDIDATES]:
        df = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            usecols=["source1_entity_id"],
        )
        ids.update(df["source1_entity_id"].dropna().tolist())

    return ids


def main():
    print("=" * 70)
    print("PHASE 3 — GENERATE SUPERVISED TRAINING PAIRS")
    print("=" * 70)

    s2 = load_candidates(S2_CANDIDATES, "S2")
    s3 = load_candidates(S3_CANDIDATES, "S3")

    candidates = pd.concat([s2, s3], ignore_index=True)

    # Remove duplicate candidate rows.
    candidates = candidates.drop_duplicates(
        subset=["source1_entity_id", "target_entity_id", "target_source"]
    )

    sampled_s1_ids = load_sampled_s1_ids()

    print(f"Candidate rows before filtering: {len(candidates):,}")
    print(f"Sampled S1 entities found: {len(sampled_s1_ids):,}")

    candidates = candidates[
        candidates["source1_entity_id"].isin(sampled_s1_ids)
    ].copy()

    candidates["y"] = 0

    gt = load_ground_truth()

    positives = gt[
        gt["source1_entity_id"].isin(sampled_s1_ids)
    ].copy()

    # Determine whether each positive target belongs to S2 or S3.
    # We do this by scanning only the ID column of each source.
    # This is reliable even when a true pair was missed by candidate retrieval.
    target_source = {}

    for source_name, path in [("S2", S2_PATH), ("S3", S3_PATH)]:
        print(f"Indexing {source_name} target IDs...")

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            usecols=["entity_id"],
            chunksize=500_000,
            keep_default_na=False,
        ):
            for entity_id in chunk["entity_id"]:
                target_source.setdefault(entity_id, source_name)

    positives["target_source"] = positives["target_entity_id"].map(target_source)

    missing_source = positives["target_source"].isna().sum()
    if missing_source:
        print(
            f"WARNING: {missing_source:,} ground-truth targets were not found "
            "in S2/S3 and will be removed."
        )

    positives = positives.dropna(subset=["target_source"])

    positives = positives[
        [
            "source1_entity_id",
            "target_entity_id",
            "target_source",
            "y",
        ]
    ].copy()

    # Add all positives, including positives that the current Top-50
    # candidate generator failed to retrieve.
    combined = pd.concat(
        [candidates, positives],
        ignore_index=True,
    )

    # If a pair appears in both candidate and ground truth, y=1 wins.
    combined["y"] = (
        combined["y"]
        .groupby(
            [
                combined["source1_entity_id"],
                combined["target_entity_id"],
                combined["target_source"],
            ]
        )
        .transform("max")
    )

    combined = combined.drop_duplicates(
        subset=[
            "source1_entity_id",
            "target_entity_id",
            "target_source",
        ]
    )

    combined = combined.sort_values(
        [
            "source1_entity_id",
            "target_source",
            "target_entity_id",
        ]
    ).reset_index(drop=True)

    combined.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False,
    )

    print()
    print(f"Positive pairs: {(combined['y'] == 1).sum():,}")
    print(f"Negative pairs: {(combined['y'] == 0).sum():,}")
    print(f"Total training pairs: {len(combined):,}")
    print(f"Saved to: {OUTPUT_PATH}")
    print()
    print("Training-pair generation complete.")


if __name__ == "__main__":
    main()
