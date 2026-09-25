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

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "match_diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# Number of Source-1 entities sampled from each match-count group.
SAMPLE_PER_GROUP = 2000

# Process large source files in chunks.
CHUNKSIZE = 200_000


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Conservative Unicode-preserving normalization.

    IMPORTANT:
    We do NOT convert Unicode to ASCII.
    """

    if text is None:
        return ""

    text = str(text)

    # Unicode canonical normalization
    text = unicodedata.normalize("NFKC", text)

    # Case normalization
    text = text.casefold()

    # Normalize ampersand
    text = text.replace("&", " and ")

    # Replace punctuation/symbols with spaces,
    # while preserving Unicode letters and numbers.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str) -> set[str]:
    """
    Unicode-aware whitespace tokenization.
    """
    normalized = normalize_text(text)

    if not normalized:
        return set()

    return set(normalized.split())


def token_jaccard(a: str, b: str) -> float:

    A = tokenize(a)
    B = tokenize(b)

    if not A and not B:
        return 1.0

    if not A or not B:
        return 0.0

    return len(A & B) / len(A | B)


def normalized_ratio(a: str, b: str) -> float:

    a_norm = normalize_text(a)
    b_norm = normalize_text(b)

    if not a_norm and not b_norm:
        return 1.0

    if not a_norm or not b_norm:
        return 0.0

    return ratio(a_norm, b_norm) / 100.0


def exact_normalized_match(a: str, b: str) -> int:

    return int(normalize_text(a) == normalize_text(b))


def raw_exact_match(a: str, b: str) -> int:

    return int(str(a).strip() == str(b).strip())


def numeric_tokens(text: str) -> set[str]:

    return set(re.findall(r"\d+", str(text)))


def numeric_overlap(a: str, b: str) -> float:

    A = numeric_tokens(a)
    B = numeric_tokens(b)

    if not A and not B:
        return 1.0

    if not A or not B:
        return 0.0

    return len(A & B) / len(A | B)


# ============================================================
# SCRIPT DETECTION
# ============================================================

def detect_scripts(text: str) -> str:

    text = str(text)

    scripts = []

    script_patterns = {
        "Latin": r"\p{Latin}",
        "Devanagari": r"\p{Devanagari}",
        "Kannada": r"\p{Kannada}",
        "Tamil": r"\p{Tamil}",
        "Telugu": r"\p{Telugu}",
        "Malayalam": r"\p{Malayalam}",
        "Bengali": r"\p{Bengali}",
        "Gujarati": r"\p{Gujarati}",
        "Gurmukhi": r"\p{Gurmukhi}",
        "Arabic": r"\p{Arabic}",
    }

    # RapidFuzz doesn't provide script detection.
    # Use Python's Unicode character names as a lightweight detector.

    for char in text:

        if not char.isalpha():
            continue

        name = unicodedata.name(char, "")

        if "LATIN" in name:
            script = "Latin"
        elif "DEVANAGARI" in name:
            script = "Devanagari"
        elif "KANNADA" in name:
            script = "Kannada"
        elif "TAMIL" in name:
            script = "Tamil"
        elif "TELUGU" in name:
            script = "Telugu"
        elif "MALAYALAM" in name:
            script = "Malayalam"
        elif "BENGALI" in name:
            script = "Bengali"
        elif "GUJARATI" in name:
            script = "Gujarati"
        elif "GURMUKHI" in name:
            script = "Gurmukhi"
        elif "ARABIC" in name:
            script = "Arabic"
        else:
            script = "Other"

        scripts.append(script)

    if not scripts:
        return "None"

    return "+".join(sorted(set(scripts)))


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("=" * 80)
print("TRUE MATCH PAIR DIAGNOSTICS")
print("=" * 80)

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype="string",
    keep_default_na=False
)

print(f"Ground truth rows: {len(gt):,}")


# ============================================================
# PARSE MATCH LIST
# ============================================================

def parse_match_ids(value: str) -> list[str]:

    if not value or str(value).strip() == "":
        return []

    return [
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    ]


gt["match_list"] = gt["matched_entity_ids"].apply(parse_match_ids)

gt["n_matches"] = gt["match_list"].str.len()


# ============================================================
# STRATIFIED SAMPLING
# ============================================================

print("\nCreating stratified sample...")

# We don't need no-match entities for pair similarity analysis.
matched_gt = gt[gt["n_matches"] > 0].copy()

# Groups:
# 1
# 2
# 3
# 4
# 5+

matched_gt["match_group"] = matched_gt["n_matches"].clip(
    upper=5
)

sampled_parts = []

for group, group_df in matched_gt.groupby("match_group"):

    n = min(SAMPLE_PER_GROUP, len(group_df))

    sampled = group_df.sample(
        n=n,
        random_state=RANDOM_SEED + int(group)
    )

    sampled_parts.append(sampled)

sampled_gt = pd.concat(
    sampled_parts,
    ignore_index=True
)

print(f"Sampled S1 entities: {len(sampled_gt):,}")


# ============================================================
# BUILD TARGET ID SETS
# ============================================================

s1_ids = set(sampled_gt["source1_entity_id"])

s2_ids = set()
s3_ids = set()

for match_list in sampled_gt["match_list"]:

    for entity_id in match_list:

        if entity_id.startswith("S2-"):
            s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            s3_ids.add(entity_id)


print(f"Target S1 IDs: {len(s1_ids):,}")
print(f"Target S2 IDs: {len(s2_ids):,}")
print(f"Target S3 IDs: {len(s3_ids):,}")


# ============================================================
# FUNCTION TO RETRIEVE ONLY TARGET RECORDS
# ============================================================

def retrieve_records(
    path: Path,
    target_ids: set[str],
    source_name: str
) -> pd.DataFrame:

    print("\n" + "-" * 80)
    print(f"Retrieving {source_name} target records")
    print(f"Target IDs: {len(target_ids):,}")
    print("-" * 80)

    found_parts = []

    remaining = set(target_ids)

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            dtype="string",
            keep_default_na=False,
            chunksize=CHUNKSIZE
        ),
        start=1
    ):

        matches = chunk[
            chunk["entity_id"].isin(remaining)
        ]

        if len(matches) > 0:

            found_parts.append(matches)

            remaining.difference_update(
                matches["entity_id"].tolist()
            )

        if chunk_number % 5 == 0:

            print(
                f"Processed chunk {chunk_number:,} | "
                f"found {len(target_ids) - len(remaining):,} | "
                f"remaining {len(remaining):,}"
            )

        if not remaining:
            print("All target records found.")
            break

    if found_parts:

        result = pd.concat(
            found_parts,
            ignore_index=True
        )

    else:

        result = pd.DataFrame(
            columns=[
                "entity_id",
                "business_name",
                "business_address",
                "country"
            ]
        )

    print(
        f"{source_name}: retrieved "
        f"{len(result):,} records"
    )

    if remaining:

        print(
            f"WARNING: {len(remaining):,} target IDs "
            f"were not found."
        )

    return result


# ============================================================
# RETRIEVE RECORDS
# ============================================================

s1_df = retrieve_records(
    S1_PATH,
    s1_ids,
    "SOURCE 1"
)

s2_df = retrieve_records(
    S2_PATH,
    s2_ids,
    "SOURCE 2"
)

s3_df = retrieve_records(
    S3_PATH,
    s3_ids,
    "SOURCE 3"
)


# ============================================================
# CREATE LOOKUP DICTIONARIES
# ============================================================

s1_lookup = s1_df.set_index("entity_id").to_dict("index")
s2_lookup = s2_df.set_index("entity_id").to_dict("index")
s3_lookup = s3_df.set_index("entity_id").to_dict("index")


# ============================================================
# BUILD TRUE MATCH PAIRS
# ============================================================

print("\nCreating pair-level diagnostics...")

records = []

for _, row in sampled_gt.iterrows():

    s1_id = row["source1_entity_id"]

    if s1_id not in s1_lookup:
        continue

    s1 = s1_lookup[s1_id]

    for matched_id in row["match_list"]:

        if matched_id.startswith("S2-"):
            target = s2_lookup.get(matched_id)
            target_source = "S2"

        elif matched_id.startswith("S3-"):
            target = s3_lookup.get(matched_id)
            target_source = "S3"

        else:
            continue

        if target is None:
            continue

        s1_name = s1["business_name"]
        s2_name = target["business_name"]

        s1_address = s1["business_address"]
        s2_address = target["business_address"]

        s1_country = s1["country"]
        s2_country = target["country"]

        records.append({

            "source1_entity_id": s1_id,

            "matched_entity_id": matched_id,

            "target_source": target_source,

            "country_same": int(
                s1_country == s2_country
            ),

            # -----------------------------
            # NAME FEATURES
            # -----------------------------

            "name_raw_exact": raw_exact_match(
                s1_name,
                s2_name
            ),

            "name_normalized_exact": exact_normalized_match(
                s1_name,
                s2_name
            ),

            "name_ratio": normalized_ratio(
                s1_name,
                s2_name
            ),

            "name_jaccard": token_jaccard(
                s1_name,
                s2_name
            ),

            "name_length_s1": len(str(s1_name)),

            "name_length_target": len(str(s2_name)),

            # -----------------------------
            # ADDRESS FEATURES
            # -----------------------------

            "s1_address_missing": int(
                str(s1_address).strip() == ""
            ),

            "target_address_missing": int(
                str(s2_address).strip() == ""
            ),

            "address_raw_exact": raw_exact_match(
                s1_address,
                s2_address
            ),

            "address_normalized_exact": exact_normalized_match(
                s1_address,
                s2_address
            ),

            "address_ratio": normalized_ratio(
                s1_address,
                s2_address
            ),

            "address_jaccard": token_jaccard(
                s1_address,
                s2_address
            ),

            "address_numeric_overlap": numeric_overlap(
                s1_address,
                s2_address
            ),

            # -----------------------------
            # SCRIPT
            # -----------------------------

            "s1_name_script": detect_scripts(
                s1_name
            ),

            "target_name_script": detect_scripts(
                s2_name
            ),
        })


diagnostics = pd.DataFrame(records)


# ============================================================
# SAVE RAW DIAGNOSTICS
# ============================================================

raw_output = OUTPUT_DIR / "true_match_pairs.csv"

diagnostics.to_csv(
    raw_output,
    index=False
)

print(
    f"\nSaved pair-level diagnostics to:\n"
    f"{raw_output}"
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("TRUE MATCH SUMMARY")
print("=" * 80)

print(f"\nTotal true pairs analyzed: {len(diagnostics):,}")


# ------------------------------------------------------------
# Overall
# ------------------------------------------------------------

numeric_columns = [
    "country_same",
    "name_raw_exact",
    "name_normalized_exact",
    "name_ratio",
    "name_jaccard",
    "address_raw_exact",
    "address_normalized_exact",
    "address_ratio",
    "address_jaccard",
    "address_numeric_overlap",
    "s1_address_missing",
    "target_address_missing",
]

print("\nOverall feature summary:")

summary = diagnostics[numeric_columns].mean().to_frame(
    "mean"
)

summary["percentage"] = summary["mean"] * 100

print(summary.to_string())


# ============================================================
# SOURCE-SPECIFIC SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("S2 VS S3 COMPARISON")
print("=" * 80)

source_summary = (
    diagnostics
    .groupby("target_source")[numeric_columns]
    .mean()
    .T
)

print(source_summary.to_string())


# ============================================================
# NAME SIMILARITY QUANTILES
# ============================================================

print("\n" + "=" * 80)
print("NAME SIMILARITY QUANTILES")
print("=" * 80)

print(
    diagnostics["name_ratio"]
    .quantile([
        0.00,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.00
    ])
)


# ============================================================
# ADDRESS SIMILARITY QUANTILES
# ============================================================

print("\n" + "=" * 80)
print("ADDRESS SIMILARITY QUANTILES")
print("=" * 80)

print(
    diagnostics["address_ratio"]
    .quantile([
        0.00,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.00
    ])
)


# ============================================================
# EXACT MATCH RATES
# ============================================================

print("\n" + "=" * 80)
print("EXACT MATCH RATES")
print("=" * 80)

print(
    f"Raw exact name rate: "
    f"{diagnostics['name_raw_exact'].mean() * 100:.2f}%"
)

print(
    f"Normalized exact name rate: "
    f"{diagnostics['name_normalized_exact'].mean() * 100:.2f}%"
)

print(
    f"Raw exact address rate: "
    f"{diagnostics['address_raw_exact'].mean() * 100:.2f}%"
)

print(
    f"Normalized exact address rate: "
    f"{diagnostics['address_normalized_exact'].mean() * 100:.2f}%"
)

print(
    f"Country agreement rate: "
    f"{diagnostics['country_same'].mean() * 100:.2f}%"
)


# ============================================================
# HIGH-SIMILARITY COVERAGE
# ============================================================

print("\n" + "=" * 80)
print("HIGH-SIMILARITY COVERAGE")
print("=" * 80)

for threshold in [0.70, 0.80, 0.85, 0.90, 0.95]:

    name_rate = (
        diagnostics["name_ratio"] >= threshold
    ).mean()

    address_rate = (
        diagnostics["address_ratio"] >= threshold
    ).mean()

    print(
        f"Threshold {threshold:.2f} | "
        f"Name: {name_rate * 100:6.2f}% | "
        f"Address: {address_rate * 100:6.2f}%"
    )


# ============================================================
# SCRIPT COMBINATIONS
# ============================================================

print("\n" + "=" * 80)
print("NAME SCRIPT COMBINATIONS")
print("=" * 80)

print(
    diagnostics[
        [
            "s1_name_script",
            "target_name_script"
        ]
    ]
    .value_counts()
    .head(20)
    .to_string()
)


print("\n" + "=" * 80)
print("DIAGNOSTICS COMPLETE")
print("=" * 80)