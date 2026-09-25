from __future__ import annotations

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "hard_negative_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_SEED = 42

# Number of S1 entities used in the experiment
N_S1_SAMPLE = 10000

# Number of hard negatives retained per S1
NEGATIVES_PER_S1 = 5

# Number of candidates temporarily retained per S1
CANDIDATES_PER_S1 = 20

CHUNKSIZE = 200_000


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

    # First 4 Unicode-aware alphanumeric characters
    compact = "".join(
        char
        for char in normalized
        if char.isalnum()
    )

    return compact[:4]


def token_set(text: str) -> set[str]:

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


def numeric_tokens(text: str) -> set[str]:

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
# LOAD GROUND TRUTH
# ============================================================

print("=" * 80)
print("HARD NEGATIVE ANALYSIS")
print("=" * 80)

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype="string",
    keep_default_na=False
)

print(
    f"Ground truth rows: {len(gt):,}"
)


# ============================================================
# PARSE TRUE MATCHES
# ============================================================

def parse_ids(value):

    if not value:
        return []

    return [
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    ]


gt["true_matches"] = (
    gt["matched_entity_ids"]
    .apply(parse_ids)
)


# ============================================================
# SAMPLE S1 ENTITIES
# ============================================================

matched_gt = gt[
    gt["true_matches"].str.len() > 0
].copy()

sample_gt = matched_gt.sample(
    n=min(
        N_S1_SAMPLE,
        len(matched_gt)
    ),
    random_state=RANDOM_SEED
)

sample_s1_ids = set(
    sample_gt["source1_entity_id"]
)

print(
    f"\nSampled S1 entities: "
    f"{len(sample_s1_ids):,}"
)


# ============================================================
# RETRIEVE S1 RECORDS
# ============================================================

print("\nRetrieving Source 1 records...")

s1_parts = []

for chunk in pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype="string",
    keep_default_na=False,
    chunksize=CHUNKSIZE
):

    found = chunk[
        chunk["entity_id"].isin(
            sample_s1_ids
        )
    ]

    if len(found):
        s1_parts.append(found)

    if sum(
        len(x)
        for x in s1_parts
    ) >= len(sample_s1_ids):

        break


s1_df = pd.concat(
    s1_parts,
    ignore_index=True
)

print(
    f"Retrieved S1 records: "
    f"{len(s1_df):,}"
)


# ============================================================
# BUILD S1 BLOCKING KEYS
# ============================================================

s1_df["name_key"] = (
    s1_df["business_name"]
    .apply(name_key)
)

s1_df["country_key"] = (
    s1_df["country"]
    .astype(str)
    .str.casefold()
)

s1_df["block_key"] = (
    s1_df["country_key"]
    + "_"
    + s1_df["name_key"]
)


# ============================================================
# CREATE LOOKUP STRUCTURES
# ============================================================

s1_lookup = (
    s1_df
    .set_index("entity_id")
    .to_dict("index")
)

s1_block_keys = set(
    s1_df["block_key"]
)

s1_id_to_block = dict(
    zip(
        s1_df["entity_id"],
        s1_df["block_key"]
    )
)


# ============================================================
# TRUE MATCH SET
# ============================================================

true_pair_set = set()

for _, row in sample_gt.iterrows():

    s1_id = row["source1_entity_id"]

    for target_id in row["true_matches"]:

        true_pair_set.add(
            (s1_id, target_id)
        )


print(
    f"Known true pairs in sample: "
    f"{len(true_pair_set):,}"
)


# ============================================================
# CANDIDATE COLLECTION
# ============================================================

def collect_candidates(
    source_path,
    source_name
):

    print("\n" + "=" * 80)
    print(
        f"Scanning {source_name} "
        f"for hard-negative candidates"
    )
    print("=" * 80)

    # Candidate lists:
    #
    # s1_id -> list of candidate records
    #
    candidate_map = {
        s1_id: []
        for s1_id in sample_s1_ids
    }

    total_candidates = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            source_path,
            sep="\t",
            dtype="string",
            keep_default_na=False,
            chunksize=CHUNKSIZE
        ),
        start=1
    ):

        # --------------------------------------------
        # Construct blocking key
        # --------------------------------------------

        chunk["name_key"] = (
            chunk["business_name"]
            .apply(name_key)
        )

        chunk["country_key"] = (
            chunk["country"]
            .astype(str)
            .str.casefold()
        )

        chunk["block_key"] = (
            chunk["country_key"]
            + "_"
            + chunk["name_key"]
        )

        relevant = chunk[
            chunk["block_key"].isin(
                s1_block_keys
            )
        ]

        if len(relevant) == 0:
            continue

        # --------------------------------------------
        # Group candidates by blocking key
        # --------------------------------------------

        grouped = relevant.groupby(
            "block_key"
        )

        for block_key, candidate_group in grouped:

            # Find S1 records belonging to this block
            matching_s1 = s1_df[
                s1_df["block_key"] == block_key
            ]

            for _, s1_row in matching_s1.iterrows():

                s1_id = s1_row["entity_id"]

                true_ids = {
                    target_id
                    for (
                        sid,
                        target_id
                    ) in true_pair_set
                    if sid == s1_id
                }

                for _, candidate in (
                    candidate_group.iterrows()
                ):

                    candidate_id = (
                        candidate["entity_id"]
                    )

                    # --------------------------------
                    # Don't accidentally create
                    # positive examples
                    # --------------------------------

                    if (
                        s1_id,
                        candidate_id
                    ) in true_pair_set:

                        continue

                    # --------------------------------
                    # Country must agree
                    # --------------------------------

                    if (
                        str(
                            candidate["country"]
                        ).casefold()
                        !=
                        str(
                            s1_row["country"]
                        ).casefold()
                    ):

                        continue

                    # --------------------------------
                    # Calculate name similarity
                    # --------------------------------

                    name_sim = similarity(
                        s1_row["business_name"],
                        candidate["business_name"]
                    )

                    # We are interested in difficult
                    # negatives, not random negatives.
                    if name_sim < 0.45:
                        continue

                    address_sim = similarity(
                        s1_row["business_address"],
                        candidate["business_address"]
                    )

                    score = (
                        0.60 * name_sim
                        +
                        0.30 * address_sim
                        +
                        0.10 * numeric_overlap(
                            s1_row["business_address"],
                            candidate["business_address"]
                        )
                    )

                    candidate_map[s1_id].append(
                        {
                            "source1_entity_id": s1_id,
                            "candidate_entity_id": candidate_id,
                            "candidate_source": source_name,

                            "name_similarity": name_sim,
                            "address_similarity": address_sim,

                            "name_jaccard": jaccard(
                                s1_row["business_name"],
                                candidate["business_name"]
                            ),

                            "address_jaccard": jaccard(
                                s1_row["business_address"],
                                candidate["business_address"]
                            ),

                            "numeric_overlap": numeric_overlap(
                                s1_row["business_address"],
                                candidate["business_address"]
                            ),

                            "country_same": 1,

                            "s1_name":
                                s1_row["business_name"],

                            "candidate_name":
                                candidate["business_name"],

                            "s1_address":
                                s1_row["business_address"],

                            "candidate_address":
                                candidate["business_address"],

                            "ranking_score": score
                        }
                    )

                    total_candidates += 1

        if chunk_number % 5 == 0:

            print(
                f"Processed chunk "
                f"{chunk_number:,} | "
                f"candidate pairs collected: "
                f"{total_candidates:,}"
            )

    return candidate_map


# ============================================================
# RUN FOR S2 AND S3
# ============================================================

s2_candidates = collect_candidates(
    S2_PATH,
    "S2"
)

s3_candidates = collect_candidates(
    S3_PATH,
    "S3"
)


# ============================================================
# SELECT HARDEST NEGATIVES
# ============================================================

print("\n" + "=" * 80)
print("SELECTING HARD NEGATIVES")
print("=" * 80)


all_negative_pairs = []

for candidate_map in [
    s2_candidates,
    s3_candidates
]:

    for s1_id, candidates in (
        candidate_map.items()
    ):

        if not candidates:
            continue

        candidates = sorted(
            candidates,
            key=lambda x: x["ranking_score"],
            reverse=True
        )

        selected = candidates[
            :NEGATIVES_PER_S1
        ]

        all_negative_pairs.extend(
            selected
        )


negative_df = pd.DataFrame(
    all_negative_pairs
)


# ============================================================
# SAVE NEGATIVES
# ============================================================

negative_path = (
    OUTPUT_DIR
    / "hard_negative_pairs.csv"
)

negative_df.to_csv(
    negative_path,
    index=False
)

print(
    f"\nHard negatives generated: "
    f"{len(negative_df):,}"
)

print(
    f"Saved to:\n"
    f"{negative_path}"
)


# ============================================================
# CREATE TRUE PAIR FEATURE DATA
# ============================================================

print("\nCreating true-pair feature data...")

true_records = []

# Retrieve target records needed for true pairs.
#
# We will build a lookup for all candidate IDs appearing
# in the sampled ground truth.

target_ids_s2 = set()
target_ids_s3 = set()

for ids in sample_gt["true_matches"]:

    for entity_id in ids:

        if entity_id.startswith("S2-"):
            target_ids_s2.add(entity_id)

        elif entity_id.startswith("S3-"):
            target_ids_s3.add(entity_id)


def retrieve_target_records(
    path,
    target_ids
):

    parts = []

    remaining = set(target_ids)

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
        chunksize=CHUNKSIZE
    ):

        found = chunk[
            chunk["entity_id"].isin(
                remaining
            )
        ]

        if len(found):

            parts.append(found)

            remaining.difference_update(
                found["entity_id"]
            )

        if not remaining:
            break

    if parts:
        return pd.concat(
            parts,
            ignore_index=True
        )

    return pd.DataFrame()


s2_true_df = retrieve_target_records(
    S2_PATH,
    target_ids_s2
)

s3_true_df = retrieve_target_records(
    S3_PATH,
    target_ids_s3
)


s2_lookup = (
    s2_true_df
    .set_index("entity_id")
    .to_dict("index")
)

s3_lookup = (
    s3_true_df
    .set_index("entity_id")
    .to_dict("index")
)


# ============================================================
# TRUE FEATURES
# ============================================================

for _, row in sample_gt.iterrows():

    s1_id = row["source1_entity_id"]

    s1 = s1_lookup.get(s1_id)

    if s1 is None:
        continue

    for target_id in row["true_matches"]:

        if target_id.startswith("S2-"):

            target = s2_lookup.get(
                target_id
            )

            source = "S2"

        else:

            target = s3_lookup.get(
                target_id
            )

            source = "S3"

        if target is None:
            continue

        true_records.append(
            {
                "source1_entity_id":+
                
                    s1_id,

                "candidate_entity_id":
                    target_id,

                "candidate_source":
                    source,

                "name_similarity":
                    similarity(
                        s1["business_name"],
                        target["business_name"]
                    ),

                "address_similarity":
                    similarity(
                        s1["business_address"],
                        target["business_address"]
                    ),

                "name_jaccard":
                    jaccard(
                        s1["business_name"],
                        target["business_name"]
                    ),

                "address_jaccard":
                    jaccard(
                        s1["business_address"],
                        target["business_address"]
                    ),

                "numeric_overlap":
                    numeric_overlap(
                        s1["business_address"],
                        target["business_address"]
                    ),

                "country_same":
                    int(
                        s1["country"]
                        ==
                        target["country"]
                    )
            }
        )


true_df = pd.DataFrame(
    true_records
)


# ============================================================
# LABEL
# ============================================================

true_df["label"] = 1

negative_df["label"] = 0


# ============================================================
# COMBINE
# ============================================================

common_columns = [
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "name_similarity",
    "address_similarity",
    "name_jaccard",
    "address_jaccard",
    "numeric_overlap",
    "country_same",
    "label"
]

true_model_df = true_df[
    common_columns
]

negative_model_df = negative_df[
    common_columns
]


model_df = pd.concat(
    [
        true_model_df,
        negative_model_df
    ],
    ignore_index=True
)


# ============================================================
# SAVE MODELING DATASET
# ============================================================

model_path = (
    OUTPUT_DIR
    / "true_vs_hard_negative.csv"
)

model_df.to_csv(
    model_path,
    index=False
)

print(
    f"\nSaved modeling dataset to:\n"
    f"{model_path}"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("TRUE VS HARD NEGATIVE SUMMARY")
print("=" * 80)

print(
    f"\nTrue pairs: "
    f"{len(true_model_df):,}"
)

print(
    f"Hard negatives: "
    f"{len(negative_model_df):,}"
)


features = [
    "name_similarity",
    "address_similarity",
    "name_jaccard",
    "address_jaccard",
    "numeric_overlap",
    "country_same"
]

print("\nFeature means:")

summary = (
    model_df
    .groupby("label")[features]
    .mean()
)

summary.index = [
    "Hard Negative",
    "True Match"
]

print(
    summary.to_string()
)


# ============================================================
# THRESHOLD SEPARATION
# ============================================================

print("\n" + "=" * 80)
print("THRESHOLD ANALYSIS")
print("=" * 80)

for feature in features[:-1]:

    print(
        f"\n--- {feature} ---"
    )

    for threshold in [
        0.50,
        0.60,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95
    ]:

        positive_rate = (
            true_model_df[feature]
            >= threshold
        ).mean()

        negative_rate = (
            negative_model_df[feature]
            >= threshold
        ).mean()

        print(
            f"{threshold:.2f} | "
            f"True: {positive_rate * 100:6.2f}% | "
            f"Hard negative: "
            f"{negative_rate * 100:6.2f}%"
        )


print("\n" + "=" * 80)
print("HARD NEGATIVE ANALYSIS COMPLETE")
print("=" * 80)