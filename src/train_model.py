from pathlib import Path
import json

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "model"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = OUTPUT_DIR / "logistic_model.joblib"
METRICS_PATH = OUTPUT_DIR / "logistic_metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 5 — MATCHING MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD FEATURES
    # --------------------------------------------------------

    print("\nLoading feature table...")

    df = pd.read_csv(
        FEATURE_PATH,
        sep="\t"
    )

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns):,}")

    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    if "y" not in df.columns:
        raise ValueError(
            "Column 'y' was not found in training_features.tsv"
        )

    print("\nLabel distribution:")

    print(
        df["y"].value_counts()
    )

    print("\nLabel proportions:")

    print(
        df["y"].value_counts(normalize=True)
    )

    # --------------------------------------------------------
    # REMOVE IDENTIFIERS / NON-NUMERIC COLUMNS
    # --------------------------------------------------------

    excluded_columns = [
        "source1_entity_id",
        "target_entity_id",
        "target_source",
        "y",
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    print("\nFeatures used by model:")

    for column in feature_columns:
        print(f"  {column}")

    print(
        f"\nNumber of model features: "
        f"{len(feature_columns)}"
    )

    X = df[feature_columns].copy()
    y = df["y"].astype(int)

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    print("\nChecking missing values...")

    missing_total = int(
        X.isna().sum().sum()
    )

    print(
        f"Total missing feature values: "
        f"{missing_total:,}"
    )

    if missing_total > 0:

        print(
            "Filling missing feature values with 0."
        )

        X = X.fillna(0)

    # Make sure every feature is numeric
    X = X.astype(float)

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    print("\nCreating train/test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"Training rows: {len(X_train):,}"
    )

    print(
        f"Testing rows:  {len(X_test):,}"
    )

    print("\nTraining labels:")

    print(
        y_train.value_counts()
    )

    print("\nTesting labels:")

    print(
        y_test.value_counts()
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 70)

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                )
            )
        ]
    )

    print("\nFitting model...")

    model.fit(
        X_train,
        y_train
    )

    print("Model training complete.")

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    average_precision = average_precision_score(
        y_test,
        y_prob
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    print(
        f"\nAccuracy:           {accuracy:.6f}"
    )

    print(
        f"Precision:          {precision:.6f}"
    )

    print(
        f"Recall:             {recall:.6f}"
    )

    print(
        f"F1 Score:           {f1:.6f}"
    )

    print(
        f"ROC-AUC:            {roc_auc:.6f}"
    )

    print(
        f"Average Precision:  {average_precision:.6f}"
    )

    print("\nConfusion matrix:")

    print(cm)

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            y_pred,
            digits=6,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    print("\nSaving model...")

    import joblib

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
        },
        MODEL_PATH
    )

    print(
        f"Model saved to:\n{MODEL_PATH}"
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics = {
        "model": "logistic_regression",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "feature_count": int(len(feature_columns)),
        "positive_training": int(y_train.sum()),
        "negative_training": int((y_train == 0).sum()),
        "positive_testing": int(y_test.sum()),
        "negative_testing": int((y_test == 0).sum()),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "confusion_matrix": cm.tolist(),
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    print(
        f"Metrics saved to:\n{METRICS_PATH}"
    )

    print("\n" + "=" * 70)
    print("PHASE 5 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()