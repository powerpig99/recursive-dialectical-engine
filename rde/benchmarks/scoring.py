"""Scoring functions for benchmark evaluation.

Matches the scoring methodology from the OOLONG benchmark paper
(Bertsch et al., 2511.02817) and the RLM paper (Zhang et al., 2512.24601).
"""

from __future__ import annotations


def oolong_accuracy(predicted: dict[str, float], ground_truth: dict[str, float]) -> float:
    """OOLONG scoring: 0.75^|y - yhat| for each category, then average.

    Both predicted and ground_truth are dicts mapping category names to counts/fractions.
    """
    if not ground_truth:
        return 0.0

    all_keys = set(ground_truth.keys()) | set(predicted.keys())
    scores = []
    for key in all_keys:
        gt_val = ground_truth.get(key, 0.0)
        pred_val = predicted.get(key, 0.0)
        scores.append(0.75 ** abs(gt_val - pred_val))

    return sum(scores) / len(scores) if scores else 0.0


def f1_pairs(
    predicted_pairs: set[tuple[str, str]],
    ground_truth_pairs: set[tuple[str, str]],
) -> float:
    """F1 score for OOLONG-Pairs: matching document pairs by topic.

    Pairs are unordered — (a, b) and (b, a) are the same match.
    """
    # Normalize pairs to canonical order
    pred_normalized = {tuple(sorted(p)) for p in predicted_pairs}
    truth_normalized = {tuple(sorted(p)) for p in ground_truth_pairs}

    if not truth_normalized and not pred_normalized:
        return 1.0
    if not truth_normalized or not pred_normalized:
        return 0.0

    true_positives = len(pred_normalized & truth_normalized)
    precision = true_positives / len(pred_normalized) if pred_normalized else 0.0
    recall = true_positives / len(truth_normalized) if truth_normalized else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(predicted: str, ground_truth: str) -> float:
    """S-NIAH scoring: case-insensitive exact match."""
    return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0
