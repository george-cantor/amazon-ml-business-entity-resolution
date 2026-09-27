# Amazon ML Challenge 2026 — Business Entity Resolution

## 1. Project Overview

This repository contains our solution for the **Amazon ML Challenge 2026 — Business Entity Resolution** problem.

The task is to identify which business records in two source datasets (`Source 2` and `Source 3`) correspond to businesses in the reference dataset (`Source 1`).

Our solution follows a two-stage entity-resolution approach:

1. Generate a manageable set of candidate pairs using blocking.
2. Rank and classify candidate pairs using a supervised machine-learning model based on name, address, country, and token-level similarity features.

The final system was designed with a strong emphasis on **precision**, since the challenge evaluates predictions using the F0.5 score.

### High-level workflow

```text
Source 1
   │
   ├── Normalization
   ├── Candidate Generation / Blocking
   ├── Top-K Candidate Selection
   ├── Feature Engineering
   ├── Logistic Regression Matching Model
   ├── Probability Threshold
   └── Final Predictions
          ├── matching_results.tsv
          └── candidate_pairs.tsv
```

---

## 2. Problem Statement

Business entity resolution is the process of determining whether records from different datasets refer to the same real-world entity.

In this challenge, `Source 1` acts as the reference dataset. For every record in `Source 1`, the goal is to identify the corresponding record or records from `Source 2` and `Source 3`.

The challenge allows zero matches, one match, or multiple matches.

---

## 3. Challenge Objective

The objective was to build a scalable matching system capable of processing millions of business records while maintaining high matching precision.

The official evaluation places greater emphasis on precision through the **F0.5 score**.

The approach therefore focuses on efficient candidate generation, informative similarity features, supervised pair classification, probability-based decision making, and validation of the exact candidate set used during final inference.

---

## 4. Dataset

```text
data/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

The training data contains known entity relationships used to construct positive and negative training pairs. The test data contains the entities for which final predictions are generated.

---

## 5. Data Sources

Source 1 is the deduplicated reference dataset.

Source 2 and Source 3 contain business records that may correspond to entities in Source 1.

The matching task is performed independently between:

```text
Source 1 ↔ Source 2
Source 1 ↔ Source 3
```

Only information supplied in the challenge datasets was used. No external business databases, web searches, geocoding services, or external entity information were used.

---

## 6. Data Exploration

The training ground truth contained:

- **2,206,821** Source 1 entities.
- **2,206,821** unique Source 1 IDs.
- **1,232,247** rows with no matched entity recorded.

After expanding the ground truth into individual positive relationships:

| Dataset | Positive Pairs |
|---|---:|
| Source 2 | 3,693,619 |
| Source 3 | 3,944,746 |
| Total | 7,638,365 |

Entity names and addresses provide useful but incomplete matching signals, so a combination of features was required.

---

## 7. Data Preprocessing

Business names were normalized using Unicode normalization, case folding, replacement of `&` with `and`, punctuation removal, whitespace normalization, trimming, and extraction of a short normalized name key.

The name key uses the first four characters of the normalized representation.

```text
Original Name
     ↓
Unicode normalization
     ↓
Case folding
     ↓
"&" → "and"
     ↓
Remove punctuation
     ↓
Normalize whitespace
     ↓
First 4 characters
     ↓
Name blocking key
```

The same normalization logic was applied consistently during training and test inference.

---

## 8. Entity Representation

Each candidate business pair was represented using:

- business name,
- address,
- country,
- missingness indicators,
- token overlap,
- character-level similarity,
- numerical token overlap.

The representation uses only information available in the challenge datasets.

---

## 9. Candidate Generation

Comparing every Source 1 record with every Source 2 and Source 3 record would produce an impractically large number of comparisons.

The initial blocking key combined:

```text
country_key + "|" + name_key
```

Address information was also used for additional candidate recovery.

On the training data:

| Blocking Strategy | Source 2 Recall | Source 3 Recall |
|---|---:|---:|
| Name | 76.7963% | 75.4399% |
| Address | 44.5090% | 43.2890% |
| Name + Address Union | 86.4082% | 86.9843% |

The union strategy recovered:

- **3,191,589** Source 2 positive pairs.
- **3,431,308** Source 3 positive pairs.

---

## 10. Blocking Strategy

Blocked candidates were ranked using inexpensive similarity information before the more expensive model-based scoring stage.

For each Source 1 entity, only the highest-ranked candidates were retained.

The final submitted configuration used:

```text
TOP_K_CANDIDATES = 10
EXACT_SCORE_K    = 10
```

This reduced the number of candidate pairs requiring full feature computation and model scoring.

---

## 11. Candidate Pair Construction

For the completed K=10 submission:

- Source 2 retained candidates: **6,530,863**
- Source 3 retained candidates: **6,548,254**
- Final candidate pairs after consolidation: **8,623,351**
- Source 1 entities with candidates: **1,378,543**

The final `candidate_pairs.tsv` corresponds to the exact candidate set passed into final model inference.

---

## 12. Feature Engineering

For every candidate pair, 18 features were calculated.

### Name features

1. `name_similarity`
2. `name_jaccard`
3. `name_token_overlap`
4. `name_containment`
5. `name_exact`
6. `name_length_ratio`

### Address features

7. `address_similarity`
8. `address_jaccard`
9. `address_token_overlap`
10. `address_containment`
11. `address_exact`
12. `address_length_ratio`

### Additional features

13. `numeric_overlap`
14. `country_same`
15. `name_missing_s1`
16. `name_missing_target`
17. `address_missing_s1`
18. `address_missing_target`

Character similarity was calculated using RapidFuzz-based comparison.

---

## 13. Matching Model

A supervised **Logistic Regression** classifier was used as the final pair-matching model.

```text
Features
   ↓
StandardScaler
   ↓
LogisticRegression
```

Configuration:

```text
class_weight = "balanced"
max_iter = 1000
random_state = 42
```

Class balancing was used because the candidate-pair dataset contains substantially more negative pairs than positive pairs.

---

## 14. Model Training

The generated training-pair dataset contained:

- **921,743** total candidate pairs.
- **34,768** positive pairs.
- **886,975** negative pairs.

The model training split contained:

- **737,394** training observations.
- **184,349** validation observations.

Model artifacts:

```text
experiments/model/logistic_model.joblib
experiments/model/logistic_metrics.json
```

---

## 15. Threshold Selection

Different probability thresholds were evaluated on the validation set.

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.10 | 0.246877 | 0.994248 | 0.395540 |
| 0.30 | 0.413946 | 0.983289 | 0.582620 |
| 0.50 | 0.523881 | 0.963789 | 0.678794 |
| 0.70 | 0.619539 | 0.929504 | 0.743509 |
| 0.80 | 0.669784 | 0.906236 | 0.770273 |
| 0.90 | 0.729900 | 0.864962 | 0.791718 |
| 0.95 | 0.769601 | 0.826910 | 0.797227 |
| 0.99 | 0.836726 | 0.697480 | 0.760784 |

Additional validation metrics:

```text
ROC-AUC = 0.992620
Average Precision = 0.854386
```

The submitted K=10 configuration used:

```text
THRESHOLD = 0.99
```

---

## 16. Final Inference

Final inference was performed over the generated candidate pairs.

Configuration:

```text
TOP_K_CANDIDATES = 10
EXACT_SCORE_K    = 10
THRESHOLD        = 0.99
```

DuckDB was used for large-scale processing.

The final inference scored:

```text
8,623,351 candidate pairs
```

The predictions contained:

- **1,053,450** Source 1 entities with at least one predicted match.
- **2,411,931** predicted entity pairs.

---

## 17. Output Generation

The final system generates:

```text
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

`matching_results.tsv` contains the final predicted matches for every Source 1 entity, including entities with no predicted match.

`candidate_pairs.tsv` contains the exact candidate pairs passed into the final matching stage.

---

## 18. Validation

The submission was validated using the validation script.

For the validated submission:

```text
Required S1 rows:       1,732,544
Matching result rows:   1,732,544
Candidate rows:         1,732,544
```

The validator returned:

```text
PASS — no blocking issues found. Safe to submit.
```

Empty predictions were represented as empty fields rather than literal `""` values.

---

## 19. Evaluation Metrics

The challenge evaluates entity resolution using the **F0.5 score**, which gives greater importance to precision than recall.

Development evaluation considered:

- precision,
- recall,
- F1,
- ROC-AUC,
- average precision,
- threshold-dependent precision/recall trade-offs.

The high decision threshold was selected to align inference with the precision-heavy evaluation objective.

---

## 20. Results

The best completed submitted configuration was:

| Parameter | Value |
|---|---:|
| Candidate K | 10 |
| Exact scoring K | 10 |
| Matching threshold | 0.99 |
| Final candidate pairs | 8,623,351 |
| S1 entities with candidates | 1,378,543 |
| S1 entities with predicted matches | 1,053,450 |
| Predicted entity pairs | 2,411,931 |

The corresponding public leaderboard score was:

```text
0.477
```

A previous K=8 configuration achieved `0.466`.

A subsequent threshold experiment using `0.995` with K=10 resulted in `0.464`.

---

## 21. Error Analysis

### Blocking errors

If the true entity is removed during candidate generation, the matching model cannot recover it later.

The name-plus-address blocking experiment improved training-set recall substantially compared with either signal individually.

### Similar business names

Business names can be shared by multiple entities, so name similarity alone is insufficient.

### Address variation

Addresses may differ because of formatting, punctuation, token ordering, abbreviations, or missing components.

### Class imbalance

The candidate-pair dataset contains many more negative examples than positive examples, motivating class balancing during model training.

### Threshold sensitivity

Increasing the probability threshold improves precision while reducing recall.

---

## 22. Lessons Learned

1. **Candidate generation is critical.** The matching model can only select among the candidates it receives.

2. **Multiple weak signals can form a strong representation.** Name, address, country, token overlap, character similarity, and missingness indicators provide complementary information.

3. **Precision and recall must be considered together.** A threshold that produces high recall is not necessarily appropriate for a precision-heavy metric.

4. **Large-scale entity resolution requires computational planning.** Efficient blocking, ranking, and columnar processing are essential.

5. **Submission validation matters.** Correct formatting and row coverage are necessary before submission.

---

## 23. Computational Considerations

The final inference pipeline uses **DuckDB** for large-scale candidate processing and scoring.

Configuration:

```text
DUCKDB_THREADS = 4
DUCKDB_MEMORY  = 5GB
```

The trained logistic-regression model is small enough to package with the project.

The design separates large-scale candidate generation from the reduced candidate set used for model scoring, avoiding construction of the full Cartesian product.

---

## 24. Reproducibility

Dependencies are pinned in:

```text
code/business_entity_resolution/requirements.txt
```

Versions used:

```text
pandas==3.0.6
numpy==2.5.3
scikit-learn==1.9.1
joblib==1.6.0
rapidfuzz==3.14.6
duckdb==1.5.5
```

The trained model is stored at:

```text
code/business_entity_resolution/models/logistic_model.joblib
```

Submission source code is contained under:

```text
code/business_entity_resolution/src/
```

---

## 25. Project Structure

```text
.
├── data/
│   ├── train/
│   └── test/
├── experiments/
│   ├── features/
│   ├── model/
│   ├── threshold_analysis/
│   └── training_pairs/
├── output/
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
├── src/
│   ├── submission.py
│   └── validate_submission.py
├── code/
│   └── business_entity_resolution/
│       ├── models/
│       │   └── logistic_model.joblib
│       ├── src/
│       ├── README.md
│       └── requirements.txt
├── Documentation_template.md
└── README.md
```

---

## 26. Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

Install dependencies:

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

---

## 27. How to Run

From the project root:

```bash
python src/submission.py
```

After generating the outputs, validate them using:

```bash
python src/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir data/test
```

Successful validation:

```text
PASS — no blocking issues found. Safe to submit.
```

---

## 28. Team

### Team Name

**Veterans**

### Team Members

- **Garg Parahar**
- **Siva Lakshmi Veesam**
- **Saurav Kumar**
- **Gagan Krishna S**

This project was developed collaboratively as part of the Amazon ML Challenge 2026.

---

## 29. Final Status

The project successfully produced the required challenge outputs and passed submission validation.

Best completed submitted configuration:

```text
Candidate K       : 10
Exact scoring K   : 10
Threshold         : 0.99
Public Score      : 0.477
```

The repository contains the implementation, trained model, requirements, generated outputs, validation utility, and project documentation.

---

## License

The machine-learning model follows the licensing requirements specified by the challenge.

The project code is provided for educational and competition purposes.
