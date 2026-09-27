from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "features"
    / "training_features.tsv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "model"
    / "logistic_model.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "threshold_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Same features used during Phase 5
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


THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99,
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PHASE 6 — THRESHOLD ANALYSIS")
print("=" * 70)

print("\nLoading feature table...")

df = pd.read_csv(
    FEATURE_PATH,
    sep="\t"
)

print(f"Rows loaded: {len(df):,}")
print(f"Columns loaded: {len(df.columns)}")


# ============================================================
# CHECK FEATURES
# ============================================================

print("\nChecking required features...")

missing_features = [
    col for col in FEATURE_COLUMNS
    if col not in df.columns
]

if missing_features:
    print("\nERROR: Missing feature columns:")
    for col in missing_features:
        print(f"  - {col}")
    raise ValueError("Required features are missing.")

print("All 18 model features found.")


# ============================================================
# PREPARE X AND Y
# ============================================================

X = df[FEATURE_COLUMNS].copy()
y = df["y"].astype(int)


print("\nLabel distribution:")
print(y.value_counts().sort_index())

print("\nLabel proportions:")
print(y.value_counts(normalize=True).sort_index())


# ============================================================
# CHECK MISSING VALUES
# ============================================================

missing_values = X.isna().sum().sum()

print(f"\nMissing feature values: {missing_values:,}")

if missing_values > 0:
    print("Filling missing feature values with 0.")
    X = X.fillna(0)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained logistic regression model...")

model_bundle = joblib.load(MODEL_PATH)

print(f"Model loaded from:")
print(MODEL_PATH)

# The saved file contains both the trained model
# and the feature-column list.
if isinstance(model_bundle, dict):

    model = model_bundle["model"]

    saved_features = model_bundle.get(
        "feature_columns",
        FEATURE_COLUMNS
    )

    print("\nSaved model bundle detected.")
    print(f"Model type: {type(model).__name__}")
    print(f"Saved feature count: {len(saved_features)}")

else:

    model = model_bundle

    print("\nDirect model object detected.")
    print(f"Model type: {type(model).__name__}")


# ============================================================
# GENERATE PROBABILITIES
# ============================================================

print("\nGenerating match probabilities...")

probabilities = model.predict_proba(X)[:, 1]

print("Probability generation complete.")

print(f"Minimum probability: {probabilities.min():.6f}")
print(f"Maximum probability: {probabilities.max():.6f}")
print(f"Mean probability:    {probabilities.mean():.6f}")


# ============================================================
# OVERALL MODEL METRICS
# ============================================================

roc_auc = roc_auc_score(y, probabilities)
average_precision = average_precision_score(y, probabilities)

print("\n" + "=" * 70)
print("OVERALL MODEL METRICS")
print("=" * 70)

print(f"ROC-AUC:          {roc_auc:.6f}")
print(f"Average Precision:{average_precision:.6f}")


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

results = []

total_rows = len(df)
total_positive = int(y.sum())

for threshold in THRESHOLDS:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    predicted_positive = int(predictions.sum())

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    candidate_rate = (
        predicted_positive / total_rows
    )

    candidates_per_positive = (
        predicted_positive / total_positive
        if total_positive > 0
        else 0
    )

    results.append({
        "threshold": threshold,
        "total_rows": total_rows,
        "true_matches": total_positive,
        "predicted_matches": predicted_positive,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "candidate_rate": candidate_rate,
        "candidates_per_true_match": candidates_per_positive,
    })

    print(
        f"\nThreshold: {threshold:.2f}"
    )

    print(
        f"  Predicted matches : {predicted_positive:,}"
    )

    print(
        f"  True positives    : {tp:,}"
    )

    print(
        f"  False positives   : {fp:,}"
    )

    print(
        f"  False negatives   : {fn:,}"
    )

    print(
        f"  Precision         : {precision:.4f}"
    )

    print(
        f"  Recall            : {recall:.4f}"
    )

    print(
        f"  F1                : {f1:.4f}"
    )

    print(
        f"  Candidate rate    : {candidate_rate:.4%}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_path = (
    OUTPUT_DIR
    / "threshold_results.tsv"
)

results_df.to_csv(
    results_path,
    sep="\t",
    index=False
)

print("\n" + "=" * 70)
print("THRESHOLD SUMMARY")
print("=" * 70)

print(
    results_df[
        [
            "threshold",
            "predicted_matches",
            "true_positives",
            "false_positives",
            "precision",
            "recall",
            "f1",
            "candidate_rate",
        ]
    ].to_string(index=False)
)


# ============================================================
# BEST F1
# ============================================================

best_f1_row = results_df.loc[
    results_df["f1"].idxmax()
]

print("\n" + "=" * 70)
print("BEST F1 THRESHOLD")
print("=" * 70)

print(
    f"Threshold : {best_f1_row['threshold']:.2f}"
)

print(
    f"Precision : {best_f1_row['precision']:.4f}"
)

print(
    f"Recall    : {best_f1_row['recall']:.4f}"
)

print(
    f"F1        : {best_f1_row['f1']:.4f}"
)

print(
    f"Candidates: {int(best_f1_row['predicted_matches']):,}"
)


# ============================================================
# HIGH-RECALL THRESHOLDS
# ============================================================

print("\n" + "=" * 70)
print("HIGH-RECALL OPTIONS")
print("=" * 70)

high_recall = results_df[
    results_df["recall"] >= 0.90
]

if len(high_recall) == 0:

    print(
        "No tested threshold achieved recall >= 90%."
    )

else:

    print(
        high_recall[
            [
                "threshold",
                "predicted_matches",
                "precision",
                "recall",
                "f1",
                "candidate_rate",
            ]
        ].to_string(index=False)
    )


# ============================================================
# RECOMMENDED WORKING THRESHOLDS
# ============================================================

print("\n" + "=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    """
Do NOT automatically choose the highest F1 threshold.

For this entity-resolution problem, we care about:

1. Recovering true matches.
2. Keeping the candidate set manageable.
3. Preserving enough recall for final ranking.

The threshold results will be used to decide the
final candidate-generation strategy.
"""
)

print("\nResults saved to:")
print(results_path)

print("\nPHASE 6 COMPLETE")
print("=" * 70)