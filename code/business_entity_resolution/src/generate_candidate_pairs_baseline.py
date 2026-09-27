import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz.fuzz import ratio


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "train"

S1_PATH = DATA_DIR / "train_source1.tsv"
S2_PATH = DATA_DIR / "train_source2.tsv"
S3_PATH = DATA_DIR / "train_source3.tsv"
GT_PATH = DATA_DIR / "train_ground_truth.tsv"

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "candidate_ranking"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# We deliberately start with 10,000 S1 entities.
# This is the same scale used in our hard-negative experiment.
N_S1_SAMPLE = 2_000

# Number of candidates retained for each S1.
TOP_K_VALUES = [5, 10, 20, 30, 50]

CHUNK_SIZE = 200_000


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:

    if text is None:
        return ""

    text = str(text)

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.casefold()

    text = text.replace(
        "&",
        " and "
    )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def name_key(name: str) -> str:

    normalized = normalize_text(name)

    if not normalized:
        return ""

    compact = "".join(
        char
        for char in normalized
        if char.isalnum()
    )

    return compact[:4]


def address_key(address: str) -> str:

    normalized = normalize_text(address)

    if not normalized:
        return ""

    # Keep alphanumeric characters.
    compact = "".join(
        char
        for char in normalized
        if char.isalnum()
    )

    # First 8 alphanumeric characters.
    #
    # This is intentionally only an additional candidate-generation
    # key. We will evaluate its recall before using it in the final
    # submission.
    return compact[:8]


# ============================================================
# SIMILARITY FEATURES
# ============================================================

def token_set(text: str) -> set:

    normalized = normalize_text(text)

    if not normalized:
        return set()

    return set(normalized.split())


def jaccard(a: str, b: str) -> float:

    A = token_set(a)
    B = token_set(b)

    if not A and not B:
        return 1.0

    if not A or not B:
        return 0.0

    return len(A & B) / len(A | B)


def numeric_tokens(text: str) -> set:

    return set(
        re.findall(
            r"\d+",
            str(text)
        )
    )


def numeric_overlap(a: str, b: str) -> float:

    A = numeric_tokens(a)
    B = numeric_tokens(b)

    if not A and not B:
        return 1.0

    if not A or not B:
        return 0.0

    return len(A & B) / len(A | B)


def similarity(a: str, b: str) -> float:

    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return ratio(a, b) / 100.0


# ============================================================
# BLOCKING KEYS
# ============================================================

def add_keys(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["country_key"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.casefold()
        .str.strip()
    )

    df["name_key"] = (
        df["business_name"]
        .fillna("")
        .astype(str)
        .map(name_key)
    )

    df["address_key"] = (
        df["business_address"]
        .fillna("")
        .astype(str)
        .map(address_key)
    )

    df["name_block"] = (
        df["country_key"]
        + "|"
        + df["name_key"]
    )

    df["address_block"] = (
        df["country_key"]
        + "|"
        + df["address_key"]
    )

    return df


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth():

    print("\nLoading ground truth...")

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print(f"Ground-truth rows: {len(gt):,}")

    true_pairs = set()

    for row in gt.itertuples(index=False):

        s1_id = row.source1_entity_id

        if not row.matched_entity_ids:
            continue

        matched_ids = str(
            row.matched_entity_ids
        ).split(",")

        for target_id in matched_ids:

            target_id = target_id.strip()

            if target_id:
                true_pairs.add(
                    (s1_id, target_id)
                )

    print(
        f"Expanded true pairs: "
        f"{len(true_pairs):,}"
    )

    return true_pairs


# ============================================================
# LOAD S1 SAMPLE
# ============================================================

def load_s1_sample():

    print("\nLoading Source 1...")

    s1 = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print(
        f"Source 1 rows: "
        f"{len(s1):,}"
    )

    sample = (
        s1
        .sample(
            n=min(
                N_S1_SAMPLE,
                len(s1)
            ),
            random_state=RANDOM_SEED
        )
        .reset_index(drop=True)
    )

    print(
        f"Sampled S1 entities: "
        f"{len(sample):,}"
    )

    return add_keys(sample)


# ============================================================
# BUILD BLOCK INDEX
# ============================================================

def build_block_index(s1):

    print("\nBuilding blocking indexes...")

    name_index = {}
    address_index = {}

    for row in s1.itertuples(index=False):

        s1_id = row.entity_id

        # Name block
        if row.name_block:

            name_index.setdefault(
                row.name_block,
                []
            ).append(s1_id)

        # Address block
        if row.address_block:

            address_index.setdefault(
                row.address_block,
                []
            ).append(s1_id)

    print(
        f"Unique name blocks: "
        f"{len(name_index):,}"
    )

    print(
        f"Unique address blocks: "
        f"{len(address_index):,}"
    )

    return name_index, address_index


# ============================================================
# S1 LOOKUP
# ============================================================

def build_s1_lookup(s1):

    return {
        row.entity_id: row
        for row in s1.itertuples(index=False)
    }


# ============================================================
# SCORE ONE CANDIDATE
# ============================================================

def score_candidate(s1_row, target_row):

    name_sim = similarity(
        s1_row.business_name,
        target_row.business_name
    )

    address_sim = similarity(
        s1_row.business_address,
        target_row.business_address
    )

    name_j = jaccard(
        s1_row.business_name,
        target_row.business_name
    )

    address_j = jaccard(
        s1_row.business_address,
        target_row.business_address
    )

    numeric = numeric_overlap(
        s1_row.business_address,
        target_row.business_address
    )

    country_same = int(
        str(s1_row.country).casefold()
        ==
        str(target_row.country).casefold()
    )

    # Preliminary ranking score.
    #
    # These are the weights from our hard-negative experiment.
    # We will NOT treat them as final until we evaluate TOP-K recall.
    score = (
        0.60 * name_sim
        + 0.30 * address_sim
        + 0.10 * numeric
    )

    return {
        "name_similarity": name_sim,
        "address_similarity": address_sim,
        "name_jaccard": name_j,
        "address_jaccard": address_j,
        "numeric_overlap": numeric,
        "country_same": country_same,
        "ranking_score": score,
    }


# ============================================================
# COLLECT CANDIDATES FROM ONE SOURCE RECORD
# ============================================================

def get_candidate_ids(
    target_row,
    name_index,
    address_index
):

    candidate_ids = set()

    # Name blocking
    if target_row.name_block:

        candidate_ids.update(
            name_index.get(
                target_row.name_block,
                []
            )
        )

    # Address blocking
    if target_row.address_block:

        candidate_ids.update(
            address_index.get(
                target_row.address_block,
                []
            )
        )

    return candidate_ids


# ============================================================
# PROCESS ONE SOURCE FILE
# ============================================================

def process_source(
    source_path,
    source_name,
    s1,
    s1_lookup,
    name_index,
    address_index,
    true_pairs
):

    print("\n" + "=" * 70)
    print(
        f"PROCESSING {source_name}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # We store only the best 50 candidates per S1.
    # TOP-K values below can then be evaluated without
    # rescanning the entire source file.
    # --------------------------------------------------------

    MAX_K = max(TOP_K_VALUES)

    best_candidates = {
        s1_id: []
        for s1_id in s1["entity_id"]
    }

    processed = 0
    relevant = 0
    candidate_pairs = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        print(
            f"Reading {source_name} "
            f"chunk {chunk_number}..."
        )

        chunk = add_keys(chunk)

        for target in chunk.itertuples(index=False):

            processed += 1

            candidate_ids = get_candidate_ids(
                target,
                name_index,
                address_index
            )

            if not candidate_ids:
                continue

            relevant += 1

            for s1_id in candidate_ids:

                s1_row = s1_lookup[s1_id]

                features = score_candidate(
                    s1_row,
                    target
                )

                candidate_pairs += 1

                record = {
                    "source1_entity_id": s1_id,
                    "target_entity_id": target.entity_id,
                    "target_source": source_name,
                    **features,
                }

                best_candidates[s1_id].append(
                    record
                )

                # Prevent an individual S1 list from
                # becoming unbounded.
                #
                # We periodically trim it.
                if len(best_candidates[s1_id]) > MAX_K * 4:

                    best_candidates[s1_id] = sorted(
                        best_candidates[s1_id],
                        key=lambda x: (
                            x["ranking_score"],
                            x["name_similarity"],
                            x["address_similarity"],
                            x["numeric_overlap"]
                        ),
                        reverse=True
                    )[:MAX_K]

        print(
            f"Processed chunk {chunk_number:,} | "
            f"source rows processed: {processed:,} | "
            f"relevant rows: {relevant:,}"
        )

    print(
        f"\nFinished {source_name}."
    )

    print(
        f"Rows processed: "
        f"{processed:,}"
    )

    print(
        f"Rows with at least one candidate: "
        f"{relevant:,}"
    )

    print(
        f"Raw candidate comparisons: "
        f"{candidate_pairs:,}"
    )

    return best_candidates


# ============================================================
# EVALUATE TOP-K
# ============================================================

def evaluate_top_k(
    best_candidates,
    true_pairs,
    source_name
):

    print("\n" + "=" * 70)
    print(
        f"TOP-K EVALUATION — {source_name}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # IMPORTANT PERFORMANCE OPTIMIZATION
    #
    # Instead of scanning ALL true_pairs for every S1 entity,
    # build a lookup once:
    #
    # true_by_s1[s1_id] = set(target_ids)
    #
    # This changes the evaluation from an extremely expensive
    # repeated scan into a simple dictionary lookup.
    # --------------------------------------------------------

    true_by_s1 = {}

    for s1_id, target_id in true_pairs:

        true_by_s1.setdefault(
            s1_id,
            set()
        ).add(target_id)

    print(
        f"Ground-truth S1 entities indexed: "
        f"{len(true_by_s1):,}"
    )

    results = []

    # --------------------------------------------------------
    # Rank candidates ONCE for each S1.
    #
    # We keep the ranked list so that TOP-5, TOP-10,
    # TOP-20, TOP-30 and TOP-50 can all be evaluated
    # without sorting repeatedly.
    # --------------------------------------------------------

    ranked_candidates = {}

    for s1_id, candidates in best_candidates.items():

        if not candidates:
            ranked_candidates[s1_id] = []
            continue

        ranked = sorted(
            candidates,
            key=lambda x: (
                x["ranking_score"],
                x["name_similarity"],
                x["address_similarity"],
                x["numeric_overlap"]
            ),
            reverse=True
        )

        ranked_candidates[s1_id] = ranked

    # --------------------------------------------------------
    # Evaluate every TOP-K
    # --------------------------------------------------------

    for k in TOP_K_VALUES:

        recovered = 0
        total_true = 0
        total_candidates = 0

        for s1_id, ranked in ranked_candidates.items():

            true_for_s1 = true_by_s1.get(
                s1_id,
                set()
            )

            total_true += len(true_for_s1)

            selected = ranked[:k]

            selected_ids = {
                candidate["target_entity_id"]
                for candidate in selected
            }

            recovered += len(
                selected_ids & true_for_s1
            )

            total_candidates += len(
                selected
            )

        recall = (
            recovered / total_true
            if total_true > 0
            else 0.0
        )

        average_candidates = (
            total_candidates
            / len(ranked_candidates)
            if ranked_candidates
            else 0.0
        )

        result = {
            "source": source_name,
            "top_k": k,
            "true_pairs": total_true,
            "recovered_true_pairs": recovered,
            "recall": recall,
            "candidate_pairs": total_candidates,
            "average_candidates_per_s1":
                average_candidates,
        }

        results.append(result)

        print(
            f"TOP-{k:2d} | "
            f"recovered: {recovered:,} / "
            f"{total_true:,} | "
            f"recall: {recall:.4%} | "
            f"candidates: {total_candidates:,} | "
            f"avg/S1: {average_candidates:.2f}"
        )

    return results

# ============================================================
# SAVE TOP-K CANDIDATES
# ============================================================

def save_candidates(
    best_candidates,
    source_name,
    k
):

    rows = []

    for s1_id, candidates in best_candidates.items():

        if not candidates:
            continue

        ranked = sorted(
            candidates,
            key=lambda x: (
                x["ranking_score"],
                x["name_similarity"],
                x["address_similarity"],
                x["numeric_overlap"]
            ),
            reverse=True
        )

        rows.extend(
            ranked[:k]
        )

    output_path = (
        OUTPUT_DIR
        / f"{source_name.lower()}_top_{k}.tsv"
    )

    df = pd.DataFrame(rows)

    df.to_csv(
        output_path,
        sep="\t",
        index=False
    )

    print(
        f"Saved: {output_path}"
    )

    print(
        f"Rows: {len(df):,}"
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 2.5 — CANDIDATE RANKING EXPERIMENT")
    print("=" * 70)

    true_pairs = load_ground_truth()

    s1 = load_s1_sample()

    name_index, address_index = (
        build_block_index(s1)
    )

    s1_lookup = build_s1_lookup(s1)

    all_results = []

    # --------------------------------------------------------
    # S2
    # --------------------------------------------------------

    s2_candidates = process_source(
        S2_PATH,
        "S2",
        s1,
        s1_lookup,
        name_index,
        address_index,
        true_pairs
    )

    s2_results = evaluate_top_k(
        s2_candidates,
        true_pairs,
        "S2"
    )

    all_results.extend(
        s2_results
    )

    # --------------------------------------------------------
    # S3
    # --------------------------------------------------------

    s3_candidates = process_source(
        S3_PATH,
        "S3",
        s1,
        s1_lookup,
        name_index,
        address_index,
        true_pairs
    )

    s3_results = evaluate_top_k(
        s3_candidates,
        true_pairs,
        "S3"
    )

    all_results.extend(
        s3_results
    )

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary = pd.DataFrame(
        all_results
    )

    summary_path = (
        OUTPUT_DIR
        / "top_k_summary.tsv"
    )

    summary.to_csv(
        summary_path,
        sep="\t",
        index=False
    )

    print("\n" + "=" * 70)
    print("TOP-K SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        f"\nSummary saved to:"
        f"\n{summary_path}"
    )

    print(
        "\nCandidate ranking experiment complete."
    )


if __name__ == "__main__":
    main()