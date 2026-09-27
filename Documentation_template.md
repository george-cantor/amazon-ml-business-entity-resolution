# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:**  Veterans 
**Team Members:** Garg Parashar, Siva Lakshmi Veesam, Saurav Kumar, Gagan Krishna S
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We used a two-stage blocking and machine-learning approach for business entity resolution. Candidate records were generated using normalized country/name and address information, reduced to a small final candidate set, and then scored using a Logistic Regression classifier trained on name, address, country, and missingness features. A precision-oriented decision threshold of 0.99 was selected from validation analysis because the challenge evaluates predictions using the precision-heavy F_0.5 metric.

---

## 2. Methodology

### 2.1 Problem Analysis

The supplied data contains business records from three independent sources with no common identifier across sources. The training data showed substantial variation in business names and addresses, including punctuation differences, abbreviations, legal-name variations, missing fields, transliteration/format differences, and other noisy representations.

The Source 1 record can have zero, one, or multiple corresponding records in Source 2 and Source 3. Therefore, the solution must handle both matching and no-match cases rather than forcing every Source 1 entity to receive a match.

During exploratory analysis, name-based blocking provided useful recall but also produced large candidate groups. Address information supplied complementary matches that were missed by name-only blocking. Combining the two blocking signals substantially improved candidate recall before the final candidate-ranking step.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier

**Core Innovation:** A lightweight, scalable hybrid pipeline that combines name- and address-based blocking with a supervised pairwise Logistic Regression model. The blocking stage reduces the search space, while the classifier uses multiple fuzzy and token-level similarity features to distinguish likely matches from non-matches.

The overall pipeline was:

1. Normalize business names and addresses.
2. Generate candidates using complementary name and address blocking.
3. Rank/filter candidates so that the final candidate set is the exact set passed to the final ML inference stage.
4. Compute pairwise similarity features.
5. Score candidate pairs using the trained Logistic Regression model.
6. Apply the validation-selected threshold of 0.99.
7. Write the required `matching_results.tsv` and `candidate_pairs.tsv` files.

---

## 3. Candidate Generation (Blocking)

The objective of blocking was to avoid comparing every Source 1 record with every Source 2/Source 3 record.

**Blocking keys used:**

- Country label.
- Normalized business-name key based on Unicode normalization, case folding, replacement of `&` with `and`, punctuation removal, whitespace normalization, and a short name prefix.
- Address-based blocking using normalized address information.
- The name and address candidate sets were combined so that a candidate could be retained through either complementary signal.

The training analysis showed the following blocking recall:

- Name-only blocking:
  - Source 2: 76.7963%
  - Source 3: 75.4399%
- Address-only blocking:
  - Source 2: 44.5090%
  - Source 3: 43.2890%
- Union of name and address blocking:
  - Source 2: 86.4082%
  - Source 3: 86.9843%

The final test candidate-generation run produced:

- Source 1 test entities: 1,732,544
- Source 2 test records: 4,887,273
- Source 3 test records: 5,082,316
- Source 2 raw blocked candidates: 11,551,155
- Source 2 retained final candidates: 5,682,280
- Source 3 raw blocked candidates: 11,635,193
- Source 3 retained final candidates: 5,702,355
- Final candidate pairs passed to ML inference: 7,390,524
- Source 1 entities with at least one candidate: 1,378,543

The final candidate list used `K = 8` candidates per source after candidate ranking. This is important because `candidate_pairs.tsv` contains the exact candidate set passed to the final ML scoring stage rather than an earlier, larger blocking output.

**How true matches were protected from being lost:**

The candidate-generation stage used complementary name and address blocking rather than relying on only one field. The training analysis demonstrated that the union recovered substantially more known training matches than either individual blocking strategy. The final candidate set was then kept small enough for scalable model inference.

---

## 4. Matching Model

### Features used:

**Name features:**

- Fuzzy name similarity (`RapidFuzz` ratio)
- Name Jaccard similarity
- Name token overlap
- Name containment
- Exact normalized-name match
- Name length ratio

**Address features:**

- Fuzzy address similarity (`RapidFuzz` ratio)
- Address Jaccard similarity
- Address token overlap
- Address containment
- Exact normalized-address match
- Address length ratio

**Other features:**

- Numeric-token overlap
- Country equality
- Source 1 name missing indicator
- Target name missing indicator
- Source 1 address missing indicator
- Target address missing indicator

A total of 18 features were used.

**Model type:** Logistic Regression with class balancing, implemented in a scikit-learn pipeline with feature standardization.

The model was trained on generated positive and negative candidate pairs. The training-pair dataset contained 921,743 pair examples, including 34,768 positive examples and 886,975 negative examples.

The validation split contained 737,394 training examples and 184,349 validation examples.

**Threshold selection method:** Validation-based threshold analysis with emphasis on the precision-heavy F_0.5 objective. The final inference threshold was set to **0.99**.

At threshold 0.99 on the pair-level validation set:

- Precision: 0.8367
- Recall: 0.6975
- F1: 0.7608

The validation ROC-AUC was approximately 0.9926 and average precision was approximately 0.8544.

---

## 5. Results & Error Analysis

### Validation results

The final pairwise validation analysis produced:

- **ROC-AUC:** 0.9926
- **Average Precision:** 0.8544
- **Precision at threshold 0.99:** 0.8367
- **Recall at threshold 0.99:** 0.6975
- **F1 at threshold 0.99:** 0.7608

These are validation metrics from the labelled training data and should not be interpreted as the hidden/public/private leaderboard score.

The validation results above are reported at the selected pairwise decision threshold of 0.99. The hidden test-set score is not available during development, so these validation metrics should not be interpreted as the official challenge leaderboard score.

### Common false positives (wrong merges)

The available validation analysis identifies false positives at the pair level, but it does not provide a sufficiently detailed categorical breakdown to claim that one particular business-name or address pattern was the dominant source of false positives. The main practical risk is that distinct businesses can share similar names or partial address information, especially when fields are incomplete or highly repetitive.

### Common false negatives (missed matches)

Similarly, the validation results quantify missed positive pairs but do not support a reliable claim about one single dominant false-negative pattern. Candidate-generation recall is an important limitation because a true pair that never enters the final candidate set cannot be recovered by the classifier.

The blocking analysis showed that combining name and address information substantially reduced this risk compared with using either blocking signal alone.

---

## 6. Conclusion

The final solution uses a scalable blocking-plus-classification pipeline designed for noisy multi-source business records. Complementary name and address blocking reduces the search space, while an 18-feature Logistic Regression model provides the final pairwise decision using a precision-oriented threshold of 0.99.

The completed pipeline generated the required full-test prediction and candidate files, and the submission files passed the provided structural validation checks.

---

## Appendix

### A. Code Artefacts

The complete code is included in the submission zip under:

`code/business_entity_resolution/`

The package contains the source files under `src/`, a `README.md` describing the pipeline and reproduction process, and a pinned `requirements.txt`.

The main inference entry point is:

`src/submission.py`

The broader source directory also contains the scripts used for candidate generation, feature generation, model training, evaluation, threshold analysis, and submission validation.

The final generated files are:

- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

The submission validator was run on these files and reported:

`PASS — no blocking issues found. Safe to submit.`

### B. Additional Results

Training and candidate-generation observations:

- Expanded labelled training pairs:
  - Source 2: 3,693,619
  - Source 3: 3,944,746
  - Total: 7,638,365
- Training-pair dataset used for model development:
  - Positive pairs: 34,768
  - Negative pairs: 886,975
  - Total: 921,743
- Final test predictions:
  - Source 1 test entities: 1,732,544
  - Source 1 entities with predicted matches: 1,025,384
  - Total predicted matched pairs: 2,256,224
- Final candidate pairs: 7,390,524

No external business/entity lookup, government business database, commercial entity-resolution API, geocoding service, or external business-data augmentation was used in the entity-resolution pipeline. The solution relies on the supplied challenge data and the submitted machine-learning pipeline.
