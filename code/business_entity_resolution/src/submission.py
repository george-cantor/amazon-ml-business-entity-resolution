from __future__ import annotations

import os
import re
import shutil
import time
import unicodedata
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio

# ============================================================
# FINAL PRODUCTION SUBMISSION PIPELINE
# ============================================================
# Design goals:
# 1. Never materialize the huge S1 x target blocking join.
# 2. Use exact-name/address matches first.
# 3. Use country-qualified prefix blocks only when their block
#    cardinality product is small enough.
# 4. Use a DuckDB/vectorized approximate model score to rank all
#    candidates, then run the EXACT training features (RapidFuzz)
#    only on a small final set.
# 5. Produce and validate both official output files.
# ============================================================

ROOT = Path(__file__).resolve().parents[3]
TEST_DIR = ROOT / "data" / "test"
MODEL_PATH = ROOT / "code" / "business_entity_resolution" / "models" / "logistic_model.joblib"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CANDIDATE_PATH = OUTPUT_DIR / "candidate_pairs.tsv"
MATCHING_PATH = OUTPUT_DIR / "matching_results.tsv"
TMP_DIR = OUTPUT_DIR / "duckdb_tmp"
TMP_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# TUNABLE LIMITS -- deliberately conservative for Windows.
# ------------------------------------------------------------
TOP_K_CANDIDATES = 8
EXACT_SCORE_K = 8

# A block is joined only if the complete block cross-product is <= this.
# This is the key protection against the previous enormous join.
MAX_BLOCK_PRODUCT = 750

# Longer keys rescue many of the large 4/8-character blocks.
MAX_LONG_BLOCK_PRODUCT = 300

# Number of DuckDB worker threads.
DUCKDB_THREADS = 4
DUCKDB_MEMORY = "5GB"
DUCKDB_TEMP_LIMIT = os.environ.get("DUCKDB_TEMP_LIMIT", "80GB")

THRESHOLD = 0.99

FEATURE_COLUMNS = [
    "name_similarity",
    "name_jaccard",
    "name_token_overlap",
    "name_containment",
    "name_exact",
    "name_length_ratio",
    "address_similarity",
    "address_jaccard",
    "address_token_overlap",
    "address_containment",
    "address_exact",
    "address_length_ratio",
    "numeric_overlap",
    "country_same",
    "name_missing_s1",
    "name_missing_target",
    "address_missing_s1",
    "address_missing_target",
]


def sql_path(p: Path) -> str:
    return str(p.resolve()).replace("'", "''")


def sql_norm(col: str) -> str:
    # Mirrors features.py normalization as closely as DuckDB SQL permits.
    return (
        "regexp_replace("
        "regexp_replace("
        f"lower(coalesce({col}, '')), "
        "'[^[:alnum:]_[:space:]]', ' ', 'g'), "
        "'\\s+', ' ', 'g'"
        ")"
    )


def sql_country(col: str) -> str:
    return f"lower(trim(coalesce({col}, '')))"


def safe_ratio_expr(a: str, b: str) -> str:
    return (
        f"CASE WHEN greatest(length({a}), length({b})) = 0 THEN 1.0 "
        f"WHEN least(length({a}), length({b})) = 0 THEN 0.0 "
        f"ELSE least(length({a}), length({b}))::DOUBLE "
        f"/ greatest(length({a}), length({b})) END"
    )


def token_list_expr(col: str) -> str:
    return f"list_distinct(string_split({col}, ' '))"


def intersection_len_expr(a: str, b: str) -> str:
    return f"length(list_intersect({a}, {b}))"


def jaccard_expr(a: str, b: str) -> str:
    A = token_list_expr(a)
    B = token_list_expr(b)
    I = intersection_len_expr(A, B)
    U = f"(length({A}) + length({B}) - {I})"
    return f"CASE WHEN {U} = 0 THEN 1.0 ELSE {I}::DOUBLE / {U} END"


def token_overlap_expr(a: str, b: str) -> str:
    A = token_list_expr(a)
    B = token_list_expr(b)
    I = intersection_len_expr(A, B)
    return f"CASE WHEN least(length({A}), length({B})) = 0 THEN 0.0 ELSE {I}::DOUBLE / least(length({A}), length({B})) END"


def containment_expr(a: str, b: str) -> str:
    A = token_list_expr(a)
    B = token_list_expr(b)
    I = intersection_len_expr(A, B)
    return (
        f"CASE WHEN length({A}) = 0 AND length({B}) = 0 THEN 1.0 "
        f"WHEN length({A}) = 0 OR length({B}) = 0 THEN 0.0 "
        f"ELSE greatest({I}::DOUBLE / length({A}), {I}::DOUBLE / length({B})) END"
    )


def numeric_tokens_expr(col: str) -> str:
    return f"list_distinct(regexp_extract_all({col}, '\\d+'))"


def numeric_overlap_expr(a: str, b: str) -> str:
    A = numeric_tokens_expr(a)
    B = numeric_tokens_expr(b)
    I = intersection_len_expr(A, B)
    U = f"(length({A}) + length({B}) - {I})"
    return f"CASE WHEN length({A}) = 0 AND length({B}) = 0 THEN 1.0 WHEN {U} = 0 THEN 0.0 ELSE {I}::DOUBLE / {U} END"


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")
bundle = joblib.load(MODEL_PATH)
if isinstance(bundle, dict):
    model = bundle["model"]
    model_features = bundle.get("feature_columns", FEATURE_COLUMNS)
else:
    model = bundle
    model_features = FEATURE_COLUMNS

print("Model loaded:", type(model).__name__)
print("Model features:", len(model_features))

# Extract the linear model so DuckDB can do a cheap approximate ranking.
# Find the scaler and logistic regression without assuming
# their Pipeline step names.

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = None
lr = None

for step_name, step_obj in model.steps:
    if isinstance(step_obj, StandardScaler):
        scaler = step_obj
    elif isinstance(step_obj, LogisticRegression):
        lr = step_obj

if scaler is None:
    raise RuntimeError(
        f"Could not find StandardScaler. "
        f"Pipeline steps: {[name for name, _ in model.steps]}"
    )

if lr is None:
    raise RuntimeError(
        f"Could not find LogisticRegression. "
        f"Pipeline steps: {[name for name, _ in model.steps]}"
    )

coef = lr.coef_[0].astype(float)
intercept = float(lr.intercept_[0])
means = scaler.mean_.astype(float)
scales = scaler.scale_.astype(float)

print("Scaler found:", type(scaler).__name__)
print("Classifier found:", type(lr).__name__)
print("Coefficients:", len(coef))

coef_map = dict(zip(model_features, coef))
mean_map = dict(zip(model_features, means))
scale_map = dict(zip(model_features, scales))


def sql_logit_expression(feature_sql: dict[str, str]) -> str:
    terms = [repr(intercept)]
    for f in model_features:
        # Use DuckDB numeric literals with enough precision.
        c = repr(float(coef_map[f]))
        mu = repr(float(mean_map[f]))
        sc = repr(float(scale_map[f]))
        terms.append(f"({c}) * (({feature_sql[f]}) - ({mu})) / ({sc})")
    return " + ".join(terms)


# ============================================================
# DUCKDB
# ============================================================

print("\nStarting DuckDB...")
con = duckdb.connect()

con.execute("SET threads = 4")
con.execute("SET memory_limit = '5GB'")
con.execute(f"SET temp_directory = '{sql_path(TMP_DIR)}'")
con.execute(f"SET max_temp_directory_size = '{DUCKDB_TEMP_LIMIT}'")
con.execute("SET preserve_insertion_order = false")

print("DuckDB threads:", 4)
print("DuckDB memory limit:", "5GB")
print("DuckDB temp directory:", TMP_DIR)
# ============================================================
# SOURCE 1
# ============================================================

print("\nLoading Source 1...")
t0 = time.time()

s1_file = sql_path(TEST_DIR / "test_source1.tsv")

con.execute(
    f"""
    CREATE TEMP TABLE s1 AS
    SELECT
        row_number() OVER () - 1 AS s1_idx,
        CAST(entity_id AS VARCHAR) AS s1_entity_id,
        {sql_norm('business_name')} AS name_norm,
        {sql_norm('business_address')} AS address_norm,
        {sql_country('country')} AS country_norm
    FROM read_csv(
        '{s1_file}', delim='\\t', header=true,
        all_varchar=true, quote='"', escape='"'
    )
    """
)

n_s1 = con.execute("SELECT COUNT(*) FROM s1").fetchone()[0]
print(f"Source 1 rows: {n_s1:,}")
print(f"Loaded Source 1 in {time.time() - t0:.1f}s")

# Multiple blocking keys. Exact keys are used separately.
con.execute(
    """
    ALTER TABLE s1 ADD COLUMN name_exact_key VARCHAR;
    ALTER TABLE s1 ADD COLUMN address_exact_key VARCHAR;
    ALTER TABLE s1 ADD COLUMN name4 VARCHAR;
    ALTER TABLE s1 ADD COLUMN name6 VARCHAR;
    ALTER TABLE s1 ADD COLUMN address8 VARCHAR;
    ALTER TABLE s1 ADD COLUMN address10 VARCHAR;
    """
)
con.execute(
    """
    UPDATE s1 SET
        name_exact_key = CASE WHEN name_norm <> '' THEN country_norm || '|' || name_norm ELSE '' END,
        address_exact_key = CASE WHEN address_norm <> '' THEN country_norm || '|' || address_norm ELSE '' END,
        name4 = CASE WHEN name_norm <> '' THEN country_norm || '|' || substr(name_norm, 1, 4) ELSE '' END,
        name6 = CASE WHEN name_norm <> '' THEN country_norm || '|' || substr(name_norm, 1, 6) ELSE '' END,
        address8 = CASE WHEN address_norm <> '' THEN country_norm || '|' || substr(address_norm, 1, 8) ELSE '' END,
        address10 = CASE WHEN address_norm <> '' THEN country_norm || '|' || substr(address_norm, 1, 10) ELSE '' END
    """
)
print("Source 1 blocking keys ready.")

# Block counts are used to reject dangerous many-to-many blocks BEFORE joining.
con.execute(
    """
    CREATE TEMP TABLE s1_counts_name4 AS
    SELECT name4 AS block, count(*)::UBIGINT AS n FROM s1 WHERE name4 <> '' GROUP BY 1;
    CREATE TEMP TABLE s1_counts_name6 AS
    SELECT name6 AS block, count(*)::UBIGINT AS n FROM s1 WHERE name6 <> '' GROUP BY 1;
    CREATE TEMP TABLE s1_counts_address8 AS
    SELECT address8 AS block, count(*)::UBIGINT AS n FROM s1 WHERE address8 <> '' GROUP BY 1;
    CREATE TEMP TABLE s1_counts_address10 AS
    SELECT address10 AS block, count(*)::UBIGINT AS n FROM s1 WHERE address10 <> '' GROUP BY 1;
    """
)


# ============================================================
# FEATURE SQL FOR APPROXIMATE RANKING
# ============================================================


def make_feature_sql(s_prefix: str, t_prefix: str) -> dict[str, str]:
    sn = f"{s_prefix}.name_norm"
    tn = f"{t_prefix}.target_name_norm"
    sa = f"{s_prefix}.address_norm"
    ta = f"{t_prefix}.target_address_norm"
    sc = f"{s_prefix}.country_norm"
    tc = f"{t_prefix}.target_country_norm"

    # Fast approximate ranking features. We intentionally avoid token-list
    # operations here because this stage can touch millions of pairs.
    # The exact 18 features are calculated only for the final shortlist.
    fast = {
        "name_similarity": f"CASE WHEN {sn} = '' OR {tn} = '' THEN 0.0 ELSE jaro_winkler_similarity({sn}, {tn}) END",
        "name_exact": f"CASE WHEN {sn} <> '' AND {sn} = {tn} THEN 1.0 ELSE 0.0 END",
        "name_length_ratio": safe_ratio_expr(sn, tn),
        "address_similarity": f"CASE WHEN {sa} = '' OR {ta} = '' THEN 0.0 ELSE jaro_winkler_similarity({sa}, {ta}) END",
        "address_exact": f"CASE WHEN {sa} <> '' AND {sa} = {ta} THEN 1.0 ELSE 0.0 END",
        "address_length_ratio": safe_ratio_expr(sa, ta),
        "numeric_overlap": numeric_overlap_expr(sa, ta),
        "country_same": f"CASE WHEN {sc} <> '' AND {tc} <> '' AND {sc} = {tc} THEN 1.0 ELSE 0.0 END",
    }

    # For features not calculated in the fast ranker, use their training
    # mean. Their standardized contribution is therefore approximately zero.
    out = {}
    for f in model_features:
        if f in fast:
            out[f] = fast[f]
        else:
            out[f] = repr(float(mean_map[f]))
    return out


# ============================================================
# TARGET SOURCE PROCESSING
# ============================================================


def process_source(filename: str, source_name: str) -> None:
    print("\n" + "=" * 72)
    print(f"Processing {source_name}")
    print("=" * 72)
    t0 = time.time()

    target_file = sql_path(TEST_DIR / filename)
    tn = sql_norm("business_name")
    ta = sql_norm("business_address")
    tc = sql_country("country")

    # Target is loaded once into DuckDB, then block counts are computed.
    con.execute(f"DROP TABLE IF EXISTS target_{source_name.lower()}")
    con.execute(
        f"""
        CREATE TEMP TABLE target_{source_name.lower()} AS
        SELECT
            CAST(entity_id AS VARCHAR) AS target_id,
            {tn} AS target_name_norm,
            {ta} AS target_address_norm,
            {tc} AS target_country_norm,
            CASE WHEN {tn} <> '' THEN {tc} || '|' || {tn} ELSE '' END AS name_exact_key,
            CASE WHEN {ta} <> '' THEN {tc} || '|' || {ta} ELSE '' END AS address_exact_key,
            CASE WHEN {tn} <> '' THEN {tc} || '|' || substr({tn},1,4) ELSE '' END AS name4,
            CASE WHEN {tn} <> '' THEN {tc} || '|' || substr({tn},1,6) ELSE '' END AS name6,
            CASE WHEN {ta} <> '' THEN {tc} || '|' || substr({ta},1,8) ELSE '' END AS address8,
            CASE WHEN {ta} <> '' THEN {tc} || '|' || substr({ta},1,10) ELSE '' END AS address10
        FROM read_csv(
            '{target_file}', delim='\\t', header=true,
            all_varchar=true, quote='"', escape='"'
        )
        """
    )

    tcount = con.execute(f"SELECT COUNT(*) FROM target_{source_name.lower()}").fetchone()[0]
    print(f"{source_name} rows: {tcount:,}")

    target = f"target_{source_name.lower()}"
    con.execute(f"DROP TABLE IF EXISTS tc4_{source_name.lower()}")
    con.execute(f"DROP TABLE IF EXISTS tc6_{source_name.lower()}")
    con.execute(f"DROP TABLE IF EXISTS ta8_{source_name.lower()}")
    con.execute(f"DROP TABLE IF EXISTS ta10_{source_name.lower()}")

    con.execute(f"CREATE TEMP TABLE tc4_{source_name.lower()} AS SELECT name4 AS block, count(*)::UBIGINT AS n FROM {target} WHERE name4 <> '' GROUP BY 1")
    con.execute(f"CREATE TEMP TABLE tc6_{source_name.lower()} AS SELECT name6 AS block, count(*)::UBIGINT AS n FROM {target} WHERE name6 <> '' GROUP BY 1")
    con.execute(f"CREATE TEMP TABLE ta8_{source_name.lower()} AS SELECT address8 AS block, count(*)::UBIGINT AS n FROM {target} WHERE address8 <> '' GROUP BY 1")
    con.execute(f"CREATE TEMP TABLE ta10_{source_name.lower()} AS SELECT address10 AS block, count(*)::UBIGINT AS n FROM {target} WHERE address10 <> '' GROUP BY 1")

    # We create candidates from six routes. The exact routes are not capped;
    # prefix routes are capped by the cross-product of the two block sizes.
    routes = []

    # Exact normalized name and exact normalized address.
    routes.append(
        f"""
        SELECT s.s1_idx, s.s1_entity_id, t.target_id,
               t.target_name_norm, t.target_address_norm, t.target_country_norm,
               1000.0 AS seed_score
        FROM s1 s
        JOIN {target} t ON s.name_exact_key <> '' AND s.name_exact_key = t.name_exact_key
        JOIN s1_counts_name4 sc ON sc.block = s.name4
        JOIN tc4_{source_name.lower()} tc ON tc.block = t.name4
        WHERE sc.n * tc.n <= {MAX_BLOCK_PRODUCT}
        """
    )
    routes.append(
        f"""
        SELECT s.s1_idx, s.s1_entity_id, t.target_id,
               t.target_name_norm, t.target_address_norm, t.target_country_norm,
               900.0 AS seed_score
        FROM s1 s
        JOIN {target} t ON s.address_exact_key <> '' AND s.address_exact_key = t.address_exact_key
        JOIN s1_counts_address8 sc ON sc.block = s.address8
        JOIN ta8_{source_name.lower()} tc ON tc.block = t.address8
        WHERE sc.n * tc.n <= {MAX_BLOCK_PRODUCT}
        """
    )
    # Cross-confirmation routes are useful for common exact names/addresses.
    routes.append(
        f"""
        SELECT s.s1_idx, s.s1_entity_id, t.target_id,
               t.target_name_norm, t.target_address_norm, t.target_country_norm,
               850.0 AS seed_score
        FROM s1 s
        JOIN {target} t
          ON s.name_exact_key <> '' AND s.name_exact_key = t.name_exact_key
         AND s.address8 <> '' AND s.address8 = t.address8
        """
    )
    routes.append(
        f"""
        SELECT s.s1_idx, s.s1_entity_id, t.target_id,
               t.target_name_norm, t.target_address_norm, t.target_country_norm,
               840.0 AS seed_score
        FROM s1 s
        JOIN {target} t
          ON s.address_exact_key <> '' AND s.address_exact_key = t.address_exact_key
         AND s.name4 <> '' AND s.name4 = t.name4
        """
    )

    # Safe short blocks.
    routes.append(
        f"""
        SELECT s.s1_idx, s.s1_entity_id, t.target_id,
               t.target_name_norm, t.target_address_norm, t.target_country_norm,
               100.0 AS seed_score
        FROM s1 s
        JOIN {target} t ON s.name4 <> '' AND s.name4 = t.name4
        JOIN s1_counts_name4 sc ON sc.block = s.name4
        JOIN tc4_{source_name.lower()} tc ON tc.block = t.name4
        WHERE sc.n * tc.n <= {MAX_BLOCK_PRODUCT}
        """
    )
    routes.append(
        f"""
        SELECT s.s1_idx, s.s1_entity_id, t.target_id,
               t.target_name_norm, t.target_address_norm, t.target_country_norm,
               90.0 AS seed_score
        FROM s1 s
        JOIN {target} t ON s.address8 <> '' AND s.address8 = t.address8
        JOIN s1_counts_address8 sc ON sc.block = s.address8
        JOIN ta8_{source_name.lower()} tc ON tc.block = t.address8
        WHERE sc.n * tc.n <= {MAX_BLOCK_PRODUCT}
        """
    )

    # Longer blocks are only used for blocks that were too large for the
    # shorter block. They add recall without opening a giant Cartesian product.
    routes.append(
        f"""
        SELECT s.s1_idx, s.s1_entity_id, t.target_id,
               t.target_name_norm, t.target_address_norm, t.target_country_norm,
               80.0 AS seed_score
        FROM s1 s
        JOIN {target} t ON s.name6 <> '' AND s.name6 = t.name6
        JOIN s1_counts_name6 sc ON sc.block = s.name6
        JOIN tc6_{source_name.lower()} tc ON tc.block = t.name6
        WHERE sc.n * tc.n <= {MAX_LONG_BLOCK_PRODUCT}
        """
    )
    routes.append(
        f"""
        SELECT s.s1_idx, s.s1_entity_id, t.target_id,
               t.target_name_norm, t.target_address_norm, t.target_country_norm,
               70.0 AS seed_score
        FROM s1 s
        JOIN {target} t ON s.address10 <> '' AND s.address10 = t.address10
        JOIN s1_counts_address10 sc ON sc.block = s.address10
        JOIN ta10_{source_name.lower()} tc ON tc.block = t.address10
        WHERE sc.n * tc.n <= {MAX_LONG_BLOCK_PRODUCT}
        """
    )

    union_sql = "\nUNION ALL\n".join(routes)

    # Deduplicate before calculating expensive text features.
    con.execute(f"DROP TABLE IF EXISTS raw_candidates_{source_name.lower()}")
    print("Generating safe blocked candidates...")
    con.execute(
        f"""
        CREATE TEMP TABLE raw_candidates_{source_name.lower()} AS
        SELECT
            s1_idx, s1_entity_id, target_id,
            max(target_name_norm) AS target_name_norm,
            max(target_address_norm) AS target_address_norm,
            max(target_country_norm) AS target_country_norm,
            max(seed_score) AS seed_score
        FROM ({union_sql}) q
        GROUP BY 1,2,3
        """
    )

    raw_n = con.execute(f"SELECT COUNT(*) FROM raw_candidates_{source_name.lower()}").fetchone()[0]
    raw_entities = con.execute(f"SELECT COUNT(DISTINCT s1_idx) FROM raw_candidates_{source_name.lower()}").fetchone()[0]
    print(f"{source_name} raw blocked candidates: {raw_n:,}")
    print(f"{source_name} S1 entities covered: {raw_entities:,}")

    # --------------------------------------------------------
    # Approximate model score in DuckDB.
    # This is only a ranking stage. Final predictions use the exact
    # RapidFuzz features in Python.
    # --------------------------------------------------------
    print(f"Ranking {source_name} candidates with vectorized model approximation...")
    fs = make_feature_sql("s", "c")
    logit = sql_logit_expression(fs)
    probability = f"1.0 / (1.0 + exp(-({logit})))"

    con.execute(f"DROP TABLE IF EXISTS ranked_{source_name.lower()}")
    con.execute(
        f"""
        CREATE TEMP TABLE ranked_{source_name.lower()} AS
        SELECT
            c.s1_idx,
            c.s1_entity_id,
            c.target_id,
            c.target_name_norm,
            c.target_address_norm,
            c.target_country_norm,
            {probability} AS approx_probability
        FROM raw_candidates_{source_name.lower()} c
        JOIN s1 s USING (s1_idx)
        """
    )

    con.execute(f"DROP TABLE raw_candidates_{source_name.lower()}")

    # Keep TOP_K for the official candidate file. Exact scoring later uses
    # EXACT_SCORE_K from these ranked candidates.
    con.execute(f"DROP TABLE IF EXISTS top_{source_name.lower()}")
    con.execute(
        f"""
        CREATE TEMP TABLE top_{source_name.lower()} AS
        SELECT *
        FROM ranked_{source_name.lower()}
        QUALIFY row_number() OVER (
            PARTITION BY s1_idx
            ORDER BY approx_probability DESC, target_id
        ) <= {TOP_K_CANDIDATES}
        """
    )
    top_n = con.execute(f"SELECT COUNT(*) FROM top_{source_name.lower()}").fetchone()[0]
    print(f"{source_name} retained top candidates: {top_n:,}")
    print(f"{source_name} completed in {time.time() - t0:.1f}s")


process_source("test_source2.tsv", "S2")
process_source("test_source3.tsv", "S3")

# ============================================================
# COMBINE S2 + S3 AND FINAL RANK
# ============================================================

print("\nCombining S2 and S3 candidates...")
t0 = time.time()

con.execute(
    f"""
    CREATE TEMP TABLE candidates AS
    SELECT *
    FROM (
        SELECT * FROM top_s2
        UNION ALL
        SELECT * FROM top_s3
    ) u
    QUALIFY row_number() OVER (
        PARTITION BY s1_idx
        ORDER BY approx_probability DESC, target_id
    ) <= {TOP_K_CANDIDATES}
    """
)

cand_n = con.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
covered = con.execute("SELECT COUNT(DISTINCT s1_idx) FROM candidates").fetchone()[0]
print(f"Final candidate pairs: {cand_n:,}")
print(f"S1 entities with candidates: {covered:,}")
print(f"Combined in {time.time() - t0:.1f}s")

# ============================================================
# WRITE candidate_pairs.tsv
# ============================================================
# ============================================================
# WRITE candidate_pairs.tsv IN CHUNKS
# ============================================================

print("\nWriting candidate_pairs.tsv in chunks...")
t0 = time.time()

import csv

WRITE_CHUNK_S1 = 50_000

# Remove an incomplete file from the failed attempt, if it exists.
if CANDIDATE_PATH.exists():
    CANDIDATE_PATH.unlink()

with open(
    CANDIDATE_PATH,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.writer(
        f,
        delimiter="\t",
        lineterminator="\n"
    )

    # Header
    writer.writerow([
        "source1_entity_id",
        "candidate_entity_ids"
    ])

    for start_idx in range(0, n_s1, WRITE_CHUNK_S1):

        end_idx = min(
            start_idx + WRITE_CHUNK_S1,
            n_s1
        )

        rows = con.execute(
            f"""
            SELECT
                s.s1_entity_id AS source1_entity_id,
                coalesce(
                    string_agg(
                        c.target_id,
                        ','
                        ORDER BY
                            c.approx_probability DESC,
                            c.target_id
                    ),
                    ''
                ) AS candidate_entity_ids

            FROM s1 s

            LEFT JOIN candidates c
                ON c.s1_idx = s.s1_idx

            WHERE
                s.s1_idx >= {start_idx}
                AND s.s1_idx < {end_idx}

            GROUP BY
                s.s1_idx,
                s.s1_entity_id

            ORDER BY
                s.s1_idx
            """
        ).fetchall()

        writer.writerows(rows)

        if (
            start_idx == 0
            or end_idx == n_s1
            or ((end_idx // WRITE_CHUNK_S1) % 5 == 0)
        ):
            print(
                f"Candidate file: "
                f"{end_idx:,}/{n_s1:,} S1 entities written "
                f"({100.0 * end_idx / n_s1:.1f}%)"
            )

print(f"Saved: {CANDIDATE_PATH}")
print(f"Written in {time.time() - t0:.1f}s")
# ============================================================
# EXACT MODEL SCORING
# ============================================================

print("\nSelecting final candidates for exact model scoring...")
con.execute(
    f"""
    CREATE TEMP TABLE score_pairs AS
    SELECT *
    FROM candidates
    QUALIFY row_number() OVER (
        PARTITION BY s1_idx
        ORDER BY approx_probability DESC, target_id
    ) <= {EXACT_SCORE_K}
    """
)

score_n = con.execute("SELECT COUNT(*) FROM score_pairs").fetchone()[0]
print(f"Pairs sent to exact RapidFuzz model: {score_n:,}")
print(f"Maximum possible: {n_s1 * EXACT_SCORE_K:,}")

# Fetch exact-score rows in chunks. We deliberately keep the chunks small
# enough for a normal Windows machine.
cur = con.execute(
    """
    SELECT
        c.s1_idx,
        c.s1_entity_id,
        s.name_norm AS s1_name,
        s.address_norm AS s1_address,
        s.country_norm AS s1_country,
        c.target_id,
        c.target_name_norm,
        c.target_address_norm,
        c.target_country_norm
    FROM score_pairs c
    JOIN s1 s USING (s1_idx)
    ORDER BY c.s1_idx, c.approx_probability DESC, c.target_id
    """
)

matches: dict[str, list[tuple[float, str]]] = {}
scored = 0
chunk_no = 0

while True:
    df = cur.fetch_df_chunk(50_000)
    if df is None or len(df) == 0:
        break

    chunk_no += 1
    feature_rows = []

    for r in df.itertuples(index=False):
        sn = r.s1_name
        tn = r.target_name_norm
        sa = r.s1_address
        ta = r.target_address_norm
        sc = r.s1_country
        tc = r.target_country_norm

        A_name = set(sn.split()) if sn else set()
        B_name = set(tn.split()) if tn else set()
        A_addr = set(sa.split()) if sa else set()
        B_addr = set(ta.split()) if ta else set()

        name_i = len(A_name & B_name)
        name_u = len(A_name | B_name)
        addr_i = len(A_addr & B_addr)
        addr_u = len(A_addr | B_addr)

        num_a = set(re.findall(r"\d+", sa)) if sa else set()
        num_b = set(re.findall(r"\d+", ta)) if ta else set()
        num_i = len(num_a & num_b)
        num_u = len(num_a | num_b)

        feature_rows.append({
            "name_similarity": ratio(sn, tn) / 100.0,
            "name_jaccard": name_i / name_u if name_u else 1.0,
            "name_token_overlap": name_i / min(len(A_name), len(B_name)) if A_name and B_name else 0.0,
            "name_containment": (max(name_i / len(A_name), name_i / len(B_name)) if A_name and B_name else (1.0 if not A_name and not B_name else 0.0)),
            "name_exact": int(bool(sn) and sn == tn),
            "name_length_ratio": (min(len(sn), len(tn)) / max(len(sn), len(tn)) if sn and tn else (1.0 if not sn and not tn else 0.0)),
            "address_similarity": ratio(sa, ta) / 100.0,
            "address_jaccard": addr_i / addr_u if addr_u else 1.0,
            "address_token_overlap": addr_i / min(len(A_addr), len(B_addr)) if A_addr and B_addr else 0.0,
            "address_containment": (max(addr_i / len(A_addr), addr_i / len(B_addr)) if A_addr and B_addr else (1.0 if not A_addr and not B_addr else 0.0)),
            "address_exact": int(bool(sa) and sa == ta),
            "address_length_ratio": (min(len(sa), len(ta)) / max(len(sa), len(ta)) if sa and ta else (1.0 if not sa and not ta else 0.0)),
            "numeric_overlap": num_i / num_u if num_u else (1.0 if not num_a and not num_b else 0.0),
            "country_same": int(bool(sc) and bool(tc) and sc == tc),
            "name_missing_s1": int(not sn),
            "name_missing_target": int(not tn),
            "address_missing_s1": int(not sa),
            "address_missing_target": int(not ta),
        })

    X = pd.DataFrame(feature_rows)[model_features]
    probs = model.predict_proba(X)[:, 1]

    for r, p in zip(df.itertuples(index=False), probs):
        p = float(p)
        if p >= THRESHOLD:
            matches.setdefault(r.s1_entity_id, []).append((p, r.target_id))

    scored += len(df)
    if chunk_no % 5 == 0 or scored == score_n:
        print(
            f"Exact ML scoring: {scored:,}/{score_n:,} "
            f"({100.0 * scored / max(score_n,1):.1f}%) | "
            f"matched S1: {len(matches):,}"
        )

# ============================================================
# WRITE matching_results.tsv
# ============================================================

print("\nWriting matching_results.tsv...")

match_map = {}
for sid, vals in matches.items():
    vals.sort(key=lambda x: (-x[0], x[1]))
    seen = set()
    ids = []
    for p, tid in vals:
        if tid not in seen:
            seen.add(tid)
            ids.append(tid)
    match_map[sid] = ",".join(ids)

con.register("match_df", pd.DataFrame({
    "source1_entity_id": list(match_map.keys()),
    "matched_entity_ids": list(match_map.values()),
}))

con.execute(
    f"""
    COPY (
        SELECT
            s.s1_entity_id AS source1_entity_id,
            coalesce(m.matched_entity_ids, '') AS matched_entity_ids
        FROM s1 s
        LEFT JOIN match_df m ON s.s1_entity_id = m.source1_entity_id
        ORDER BY s.s1_idx
    ) TO '{sql_path(MATCHING_PATH)}'
    WITH (HEADER TRUE, DELIMITER '\\t')
    """
)

# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 72)
print("VALIDATING FINAL OUTPUT")
print("=" * 72)

cand = pd.read_csv(CANDIDATE_PATH, sep="\t", dtype=str, keep_default_na=False)
mat = pd.read_csv(MATCHING_PATH, sep="\t", dtype=str, keep_default_na=False)

errors = []

if list(cand.columns) != ["source1_entity_id", "candidate_entity_ids"]:
    errors.append("candidate_pairs.tsv has incorrect columns")
if list(mat.columns) != ["source1_entity_id", "matched_entity_ids"]:
    errors.append("matching_results.tsv has incorrect columns")
if len(cand) != n_s1:
    errors.append(f"candidate_pairs rows = {len(cand):,}, expected {n_s1:,}")
if len(mat) != n_s1:
    errors.append(f"matching_results rows = {len(mat):,}, expected {n_s1:,}")
if cand["source1_entity_id"].duplicated().any():
    errors.append("duplicate Source 1 IDs in candidate_pairs.tsv")
if mat["source1_entity_id"].duplicated().any():
    errors.append("duplicate Source 1 IDs in matching_results.tsv")

candidate_map = dict(zip(cand.source1_entity_id, cand.candidate_entity_ids))
match_map_out = dict(zip(mat.source1_entity_id, mat.matched_entity_ids))

invalid_prediction_ids = 0
for sid, value in match_map_out.items():
    if not value:
        continue
    candidates = set(candidate_map.get(sid, "").split(",")) if candidate_map.get(sid, "") else set()
    for tid in value.split(","):
        if tid not in candidates:
            invalid_prediction_ids += 1

if invalid_prediction_ids:
    errors.append(f"{invalid_prediction_ids:,} predicted IDs are not in candidate lists")

candidate_nonempty = int((cand.candidate_entity_ids != "").sum())
matched_nonempty = int((mat.matched_entity_ids != "").sum())
total_matches = sum(len(x.split(",")) for x in mat.matched_entity_ids if x)

print(f"Source 1 rows             : {n_s1:,}")
print(f"Candidate rows             : {len(cand):,}")
print(f"S1 with candidates         : {candidate_nonempty:,}")
print(f"Matching rows              : {len(mat):,}")
print(f"S1 with matches            : {matched_nonempty:,}")
print(f"Total predicted pairs      : {total_matches:,}")
print(f"Threshold                  : {THRESHOLD}")
print(f"Candidate K                : {TOP_K_CANDIDATES}")
print(f"Exact model-score K        : {EXACT_SCORE_K}")

if errors:
    print("\nVALIDATION FAILED")
    for e in errors:
        print(" -", e)
    raise RuntimeError("Submission validation failed. Do not upload the files.")

print("\nVALIDATION PASSED")
print("\nFINAL FILES:")
print(CANDIDATE_PATH)
print(MATCHING_PATH)
print("\nFINAL SUBMISSION PIPELINE COMPLETE")
