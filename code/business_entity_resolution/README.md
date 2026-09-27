# Amazon ML Challenge 2026 — Business Entity Resolution

## Overview

This project solves the Amazon ML Challenge 2026 business entity resolution task.

The goal is to identify Source 2 and Source 3 business records that correspond to each Source 1 business record.

The solution uses:

1. Text normalization
2. Candidate blocking
3. Candidate ranking
4. Feature engineering
5. Logistic Regression
6. Probability thresholding
7. Final submission generation

## Data

The expected data structure is:

data/
+-- train/
¦   +-- train_source1.tsv
¦   +-- train_source2.tsv
¦   +-- train_source3.tsv
¦   +-- train_ground_truth.tsv
+-- test/
    +-- test_source1.tsv
    +-- test_source2.tsv
    +-- test_source3.tsv

Each source contains:

- entity_id
- business_name
- business_address
- country

## Normalization

Business names and addresses are normalized using Unicode normalization, case folding, punctuation removal, whitespace normalization, and related text cleaning.

Country values are normalized separately.

The normalization is designed to preserve multilingual text while reducing differences caused by capitalization, punctuation, and formatting.

## Candidate Generation

Candidate generation is performed using blocking based on normalized business names and addresses together with country information.

The purpose of blocking is to avoid comparing every Source 1 record against every Source 2 and Source 3 record.

Candidate blocks are protected with size limits to avoid extremely large Cartesian products.

Candidates are subsequently ranked using an approximate model score.

The final candidate set contains at most 8 candidates per Source 1 entity.

## Feature Engineering

The final matching model uses 18 features:

1. name_similarity
2. name_jaccard
3. name_token_overlap
4. name_containment
5. name_exact
6. name_length_ratio
7. address_similarity
8. address_jaccard
9. address_token_overlap
10. address_containment
11. address_exact
12. address_length_ratio
13. numeric_overlap
14. country_same
15. name_missing_s1
16. name_missing_target
17. address_missing_s1
18. address_missing_target

The similarity features use RapidFuzz.

## Model

The final matching model is Logistic Regression with StandardScaler preprocessing.

The model was trained using labeled candidate pairs.

The trained model is stored as:

models/logistic_model.joblib

## Decision Rule

A candidate is predicted as a match when:

P(match) >= 0.99

The threshold was selected using validation data with the challenge F0.5 metric in mind.

## Final Candidate Set

The final candidate set and exact model scoring use the same value:

TOP_K_CANDIDATES = 8
EXACT_SCORE_K = 8

Therefore, candidate_pairs.tsv contains exactly the candidate records supplied to the final matching model.

## Output Files

The final submission produces:

output/matching_results.tsv

with columns:

source1_entity_id
matched_entity_ids

and:

output/candidate_pairs.tsv

with columns:

source1_entity_id
candidate_entity_ids

Every Source 1 test entity is represented exactly once.

## Reproduction

Install dependencies:

    pip install -r requirements.txt

Run the final submission pipeline from the project root:

    python src/submission.py

The pipeline reads the test data and trained model and writes the final submission files to:

    output/candidate_pairs.tsv
    output/matching_results.tsv

The included validation script can be run with:

    python src/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/test

## Validation

The final generated submission was checked using the challenge validation script.

The final validation result was:

PASS — no blocking issues found. Safe to submit.

## Restrictions

No external business/entity database, commercial entity lookup, geocoding service, or online business lookup was used for entity matching.

The matching process relies on the supplied challenge data and the trained machine-learning model.
