from __future__ import annotations

from typing import Iterable, Set


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """
    Compute F_beta from precision and recall.
    beta < 1 gives more weight to precision.
    """
    if precision + recall == 0:
        return 0.0

    beta_squared = beta ** 2

    return (
        (1 + beta_squared)
        * precision
        * recall
        / (beta_squared * precision + recall)
    )


def entity_f05(
    predicted: Iterable[str],
    truth: Iterable[str],
) -> float:
    """
    Compute F_0.5 for one Source-1 entity.
    """

    predicted_set: Set[str] = set(predicted)
    truth_set: Set[str] = set(truth)

    # Special singleton/no-match case
    if len(truth_set) == 0:
        return 1.0 if len(predicted_set) == 0 else 0.0

    true_positive = len(predicted_set & truth_set)
    false_positive = len(predicted_set - truth_set)
    false_negative = len(truth_set - predicted_set)

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    precision = (
        true_positive / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        true_positive / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    return f_beta(precision, recall, beta=0.5)

#Add the macro-F₀.₅ evaluator
def macro_f05(
    predictions: dict[str, Iterable[str]],
    ground_truth: dict[str, Iterable[str]],
) -> float:
    """
    Compute macro F_0.5 over Source-1 entities.
    """

    scores = []

    for source1_id, truth in ground_truth.items():
        predicted = predictions.get(source1_id, [])

        score = entity_f05(
            predicted=predicted,
            truth=truth,
        )

        scores.append(score)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)