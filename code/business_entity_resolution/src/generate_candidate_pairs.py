import heapq
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

# Keep this separate from the old experiment so nothing is overwritten.
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "candidate_ranking_fast"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# Start with 2,000. Change to 10_000 only after the test completes.
N_S1_SAMPLE = 10_000

# Values evaluated from the same Top-50 working set.
TOP_K_VALUES = [5, 10, 20, 30, 50]
MAX_K = max(TOP_K_VALUES)

# TSV read size. The source files are large, so we stream them.
CHUNK_SIZE = 200_000



# ============================================================
# TEXT NORMALIZATION
# ============================================================


def normalize_text(text: str) -> str:
    """Normalize one text value. Used mainly for small S1-side operations."""
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def name_key(name: str) -> str:
    normalized = normalize_text(name)
    if not normalized:
        return ""

    compact = "".join(char for char in normalized if char.isalnum())
    return compact[:4]


def address_key(address: str) -> str:
    normalized = normalize_text(address)
    if not normalized:
        return ""

    compact = "".join(char for char in normalized if char.isalnum())
    return compact[:8]


def numeric_token_set(text: str) -> frozenset:
    """Precompute numeric tokens once per row instead of once per candidate pair."""
    if not text:
        return frozenset()
    return frozenset(re.findall(r"\d+", text))


# ============================================================
# VECTORIZED PREPROCESSING
# ============================================================


def normalized_series(series: pd.Series) -> pd.Series:
    """Vectorized equivalent of normalize_text for a pandas Series."""
    return (
        series.fillna("")
        .astype(str)
        .str.normalize("NFKC")
        .str.casefold()
        .str.replace("&", " and ", regex=False)
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def first_alnum(series: pd.Series, n: int) -> pd.Series:
    """Return first n alphanumeric characters from already-normalized text."""
    return (
        series.str.replace(r"[^\w]", "", regex=True)
        .str.slice(0, n)
    )


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add all fields needed by blocking and ranking.

    Expensive text normalization is performed once per row rather than once
    for every candidate comparison.
    """
    result = df.copy()

    result["country_key"] = (
        result["country"].fillna("").astype(str).str.casefold().str.strip()
    )

    result["name_norm"] = normalized_series(result["business_name"])
    result["address_norm"] = normalized_series(result["business_address"])

    result["name_key"] = first_alnum(result["name_norm"], 4)
    result["address_key"] = first_alnum(result["address_norm"], 8)

    result["name_block"] = result["country_key"] + "|" + result["name_key"]
    result["address_block"] = result["country_key"] + "|" + result["address_key"]

    # Numeric tokens are needed by the ranking score. Compute them once per row.
    result["numeric_tokens"] = result["address_norm"].map(numeric_token_set)

    return result


# ============================================================
# GROUND TRUTH
# ============================================================


def load_ground_truth():
    print("\nLoading ground truth...")

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(f"Ground-truth rows: {len(gt):,}")

    true_pairs = set()

    for row in gt.itertuples(index=False):
        s1_id = row.source1_entity_id

        if not row.matched_entity_ids:
            continue

        for target_id in str(row.matched_entity_ids).split(","):
            target_id = target_id.strip()
            if target_id:
                true_pairs.add((s1_id, target_id))

    print(f"Expanded true pairs: {len(true_pairs):,}")
    return true_pairs


def build_true_by_s1(true_pairs):
    """Build once and reuse for S2 and S3 evaluation."""
    true_by_s1 = {}

    for s1_id, target_id in true_pairs:
        true_by_s1.setdefault(s1_id, set()).add(target_id)

    print(f"Ground-truth S1 entities indexed: {len(true_by_s1):,}")
    return true_by_s1


# ============================================================
# LOAD S1 SAMPLE
# ============================================================


def load_s1_sample():
    print("\nLoading Source 1...")

    s1 = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(f"Source 1 rows: {len(s1):,}")

    sample = (
        s1.sample(
            n=min(N_S1_SAMPLE, len(s1)),
            random_state=RANDOM_SEED,
        )
        .reset_index(drop=True)
    )

    print(f"Sampled S1 entities: {len(sample):,}")
    return add_keys(sample)


# ============================================================
# BLOCK INDEX
# ============================================================


def build_block_index(s1):
    print("\nBuilding blocking indexes...")

    name_index = {}
    address_index = {}

    for row in s1.itertuples(index=False):
        s1_id = row.entity_id

        if row.name_block:
            name_index.setdefault(row.name_block, []).append(s1_id)

        if row.address_block:
            address_index.setdefault(row.address_block, []).append(s1_id)

    print(f"Unique name blocks: {len(name_index):,}")
    print(f"Unique address blocks: {len(address_index):,}")

    return name_index, address_index


def build_s1_lookup(s1):
    return {row.entity_id: row for row in s1.itertuples(index=False)}


# ============================================================
# CANDIDATE RETRIEVAL
# ============================================================


def get_candidate_ids(target_row, name_index, address_index):
    """Union name and address blocking candidates."""
    candidate_ids = set()

    if target_row.name_block:
        candidate_ids.update(name_index.get(target_row.name_block, ()))

    if target_row.address_block:
        candidate_ids.update(address_index.get(target_row.address_block, ()))

    return candidate_ids


# ============================================================
# FAST FEATURE CALCULATIONS
# ============================================================


def jaccard_from_normalized(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    A = set(a.split())
    B = set(b.split())
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0

    return len(A & B) / len(A | B)


def numeric_overlap_from_sets(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def ranking_features(s1_row, target_row, calculate_diagnostics=False):
    """Calculate only the features needed for ranking.

    Name/address strings are already normalized, so no repeated normalization
    is performed here.
    """
    name_sim = ratio(s1_row.name_norm, target_row.name_norm) / 100.0
    address_sim = ratio(s1_row.address_norm, target_row.address_norm) / 100.0
    numeric = numeric_overlap_from_sets(
        s1_row.numeric_tokens,
        target_row.numeric_tokens,
    )

    score = (
        0.60 * name_sim
        + 0.30 * address_sim
        + 0.10 * numeric
    )

    if not calculate_diagnostics:
        return score, name_sim, address_sim, numeric

    name_j = jaccard_from_normalized(s1_row.name_norm, target_row.name_norm)
    address_j = jaccard_from_normalized(
        s1_row.address_norm,
        target_row.address_norm,
    )

    country_same = int(s1_row.country_key == target_row.country_key)

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
# TOP-K HEAP
# ============================================================


def heap_key(score, name_sim, address_sim, numeric, counter):
    """Ascending key: the smallest item is the current worst candidate."""
    return (score, name_sim, address_sim, numeric, counter)


def maybe_add_top_candidate(heap, record, counter):
    """Keep only the best MAX_K candidates for one S1 entity."""
    key = heap_key(
        record["ranking_score"],
        record["name_similarity"],
        record["address_similarity"],
        record["numeric_overlap"],
        counter,
    )

    item = (key, record)

    if len(heap) < MAX_K:
        heapq.heappush(heap, item)
    elif key > heap[0][0]:
        heapq.heapreplace(heap, item)


def heap_to_sorted_records(heap):
    records = [item[1] for item in heap]
    records.sort(
        key=lambda x: (
            x["ranking_score"],
            x["name_similarity"],
            x["address_similarity"],
            x["numeric_overlap"],
        ),
        reverse=True,
    )
    return records


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
):
    print("\n" + "=" * 70)
    print(f"PROCESSING {source_name}")
    print("=" * 70)

    # Each S1 gets at most MAX_K records in memory.
    heaps = {s1_id: [] for s1_id in s1["entity_id"]}

    processed = 0
    relevant = 0
    candidate_pairs = 0
    counter = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        print(f"Reading {source_name} chunk {chunk_number}...")

        chunk = add_keys(chunk)

        for target in chunk.itertuples(index=False):
            processed += 1

            candidate_ids = get_candidate_ids(
                target,
                name_index,
                address_index,
            )

            if not candidate_ids:
                continue

            relevant += 1

            target_numeric = target.numeric_tokens

            for s1_id in candidate_ids:
                s1_row = s1_lookup[s1_id]
                heap = heaps[s1_id]

                # ----------------------------------------------------
                # Safe upper-bound pruning.
                # score <= 0.60*name + 0.30*address + 0.10*1
                # If even the maximum possible score cannot beat the
                # current worst Top-K candidate, skip this pair.
                # ----------------------------------------------------
                name_sim = ratio(
                    s1_row.name_norm,
                    target.name_norm,
                ) / 100.0

                if len(heap) >= MAX_K:
                    worst_score = heap[0][0][0]
                    max_possible = 0.60 * name_sim + 0.40
                    if max_possible <= worst_score:
                        candidate_pairs += 1
                        counter += 1
                        continue

                address_sim = ratio(
                    s1_row.address_norm,
                    target.address_norm,
                ) / 100.0

                if len(heap) >= MAX_K:
                    worst_score = heap[0][0][0]
                    max_possible = (
                        0.60 * name_sim
                        + 0.30 * address_sim
                        + 0.10
                    )
                    if max_possible <= worst_score:
                        candidate_pairs += 1
                        counter += 1
                        continue

                numeric = numeric_overlap_from_sets(
                    s1_row.numeric_tokens,
                    target_numeric,
                )

                score = (
                    0.60 * name_sim
                    + 0.30 * address_sim
                    + 0.10 * numeric
                )

                candidate_pairs += 1
                counter += 1

                record = {
                    "source1_entity_id": s1_id,
                    "target_entity_id": target.entity_id,
                    "target_source": source_name,
                    "name_similarity": name_sim,
                    "address_similarity": address_sim,
                    "numeric_overlap": numeric,
                    "ranking_score": score,
                }

                maybe_add_top_candidate(
                    heap,
                    record,
                    counter,
                )

        print(
            f"Processed chunk {chunk_number:,} | "
            f"source rows processed: {processed:,} | "
            f"relevant rows: {relevant:,} | "
            f"candidate comparisons: {candidate_pairs:,}"
        )

    print(f"\nFinished {source_name}.")
    print(f"Rows processed: {processed:,}")
    print(f"Rows with at least one candidate: {relevant:,}")
    print(f"Candidate comparisons: {candidate_pairs:,}")

    # Convert heaps to the same logical structure used by the evaluator.
    best_candidates = {
        s1_id: heap_to_sorted_records(heap)
        for s1_id, heap in heaps.items()
    }

    retained = sum(len(v) for v in best_candidates.values())
    print(f"Top-{MAX_K} candidates retained in memory: {retained:,}")

    return best_candidates


# ============================================================
# TOP-K EVALUATION
# ============================================================


def evaluate_top_k(best_candidates, true_by_s1, source_name):
    print("\n" + "=" * 70)
    print(f"TOP-K EVALUATION — {source_name}")
    print("=" * 70)

    results = []

    # Candidates are already sorted by process_source().
    for k in TOP_K_VALUES:
        recovered = 0
        total_true = 0
        total_candidates = 0

        for s1_id, ranked in best_candidates.items():
            true_for_s1 = true_by_s1.get(s1_id, set())
            total_true += len(true_for_s1)

            selected = ranked[:k]
            selected_ids = {
                candidate["target_entity_id"]
                for candidate in selected
            }

            recovered += len(selected_ids & true_for_s1)
            total_candidates += len(selected)

        recall = recovered / total_true if total_true > 0 else 0.0
        average_candidates = (
            total_candidates / len(best_candidates)
            if best_candidates
            else 0.0
        )

        result = {
            "source": source_name,
            "top_k": k,
            "true_pairs": total_true,
            "recovered_true_pairs": recovered,
            "recall": recall,
            "candidate_pairs": total_candidates,
            "average_candidates_per_s1": average_candidates,
        }
        results.append(result)

        print(
            f"TOP-{k:2d} | "
            f"recovered: {recovered:,} / {total_true:,} | "
            f"recall: {recall:.4%} | "
            f"candidates: {total_candidates:,} | "
            f"avg/S1: {average_candidates:.2f}"
        )

    return results


# ============================================================
# SAVE TOP-K CANDIDATES
# ============================================================


def save_candidates(best_candidates, source_name, k):
    rows = []

    for candidates in best_candidates.values():
        if candidates:
            rows.extend(candidates[:k])

    output_path = OUTPUT_DIR / f"{source_name.lower()}_top_{k}.tsv"

    pd.DataFrame(rows).to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print(f"Saved: {output_path}")
    print(f"Rows: {len(rows):,}")
    return output_path


# ============================================================
# MAIN
# ============================================================


def main():
    print("=" * 70)
    print("PHASE 2.5 — OPTIMIZED CANDIDATE RANKING EXPERIMENT")
    print("=" * 70)
    print(f"S1 sample: {N_S1_SAMPLE:,}")
    print(f"Top-K working set: {MAX_K}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    true_pairs = load_ground_truth()
    true_by_s1 = build_true_by_s1(true_pairs)

    s1 = load_s1_sample()
    name_index, address_index = build_block_index(s1)
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
    )

    s2_results = evaluate_top_k(
        s2_candidates,
        true_by_s1,
        "S2",
    )
    all_results.extend(s2_results)

    for k in TOP_K_VALUES:
        save_candidates(s2_candidates, "S2", k)

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
    )

    s3_results = evaluate_top_k(
        s3_candidates,
        true_by_s1,
        "S3",
    )
    all_results.extend(s3_results)

    for k in TOP_K_VALUES:
        save_candidates(s3_candidates, "S3", k)

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary = pd.DataFrame(all_results)
    summary_path = OUTPUT_DIR / "top_k_summary.tsv"

    summary.to_csv(
        summary_path,
        sep="\t",
        index=False,
    )

    print("\n" + "=" * 70)
    print("TOP-K SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))
    print(f"\nSummary saved to:\n{summary_path}")
    print("\nOptimized candidate ranking experiment complete.")


if __name__ == "__main__":
    main()
