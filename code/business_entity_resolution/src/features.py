"""
Feature engineering for business-entity matching.

This module intentionally keeps raw values separate from normalized values.
It creates pairwise features that can later be used by Logistic Regression
or a tree-based model.
"""

import re
import unicodedata
import pandas as pd
from rapidfuzz.fuzz import ratio


def normalize_text(text):
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def token_set(text):
    if not text:
        return set()
    return set(text.split())


def numeric_token_set(text):
    if not text:
        return set()
    return set(re.findall(r"\d+", text))


def jaccard(a, b):
    A = token_set(a)
    B = token_set(b)

    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0

    return len(A & B) / len(A | B)


def token_overlap(a, b):
    A = token_set(a)
    B = token_set(b)

    if not A or not B:
        return 0.0

    return len(A & B) / min(len(A), len(B))


def containment(a, b):
    A = token_set(a)
    B = token_set(b)

    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0

    return max(
        len(A & B) / len(A),
        len(A & B) / len(B),
    )


def numeric_overlap(a, b):
    A = numeric_token_set(a)
    B = numeric_token_set(b)

    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0

    return len(A & B) / len(A | B)


def length_ratio(a, b):
    la = len(a)
    lb = len(b)

    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0

    return min(la, lb) / max(la, lb)


def exact_match(a, b):
    return int(bool(a) and a == b)


def build_pair_features(s1_row, target_row):
    """
    s1_row and target_row must provide:
        business_name
        business_address
        country
    """

    s1_name = normalize_text(s1_row["business_name"])
    target_name = normalize_text(target_row["business_name"])

    s1_address = normalize_text(s1_row["business_address"])
    target_address = normalize_text(target_row["business_address"])

    s1_country = str(s1_row["country"]).casefold().strip()
    target_country = str(target_row["country"]).casefold().strip()

    name_sim = ratio(s1_name, target_name) / 100.0
    address_sim = ratio(s1_address, target_address) / 100.0

    return {
        # Name
        "name_similarity": name_sim,
        "name_jaccard": jaccard(s1_name, target_name),
        "name_token_overlap": token_overlap(s1_name, target_name),
        "name_containment": containment(s1_name, target_name),
        "name_exact": exact_match(s1_name, target_name),
        "name_length_ratio": length_ratio(s1_name, target_name),

        # Address
        "address_similarity": address_sim,
        "address_jaccard": jaccard(s1_address, target_address),
        "address_token_overlap": token_overlap(
            s1_address, target_address
        ),
        "address_containment": containment(
            s1_address, target_address
        ),
        "address_exact": exact_match(
            s1_address, target_address
        ),
        "address_length_ratio": length_ratio(
            s1_address, target_address
        ),

        # Numeric/address information
        "numeric_overlap": numeric_overlap(
            s1_address, target_address
        ),

        # Country
        "country_same": int(
            bool(s1_country)
            and bool(target_country)
            and s1_country == target_country
        ),

        # Missingness
        "name_missing_s1": int(not s1_name),
        "name_missing_target": int(not target_name),
        "address_missing_s1": int(not s1_address),
        "address_missing_target": int(not target_address),
    }


def build_feature_table(pair_df, s1_df, s2_df, s3_df):
    """
    Convert a training-pair table into an ML feature table.

    pair_df columns:
        source1_entity_id
        target_entity_id
        target_source
        y

    s1_df, s2_df, s3_df must contain:
        entity_id, business_name, business_address, country
    """

    s1_lookup = (
        s1_df.set_index("entity_id")
        [["business_name", "business_address", "country"]]
        .to_dict("index")
    )

    s2_lookup = (
        s2_df.set_index("entity_id")
        [["business_name", "business_address", "country"]]
        .to_dict("index")
    )

    s3_lookup = (
        s3_df.set_index("entity_id")
        [["business_name", "business_address", "country"]]
        .to_dict("index")
    )

    rows = []

    for row in pair_df.itertuples(index=False):
        s1_id = row.source1_entity_id
        target_id = row.target_entity_id

        s1_row = s1_lookup.get(s1_id)

        if row.target_source == "S2":
            target_row = s2_lookup.get(target_id)
        else:
            target_row = s3_lookup.get(target_id)

        if s1_row is None or target_row is None:
            continue

        features = build_pair_features(s1_row, target_row)

        rows.append(
            {
                "source1_entity_id": s1_id,
                "target_entity_id": target_id,
                "target_source": row.target_source,
                "y": row.y,
                **features,
            }
        )

    return pd.DataFrame(rows)
