"""Scoring functions for benchmark evaluation.

OOLONG scoring matches the official implementation from Bertsch et al. (2511.02817):
- NUMERIC: 0.75^|gold - predicted| (exponential decay partial credit)
- LABEL, USER, MONTH_YEAR: exact string match
- COMPARISON: substring match ("more common" in "more common than")
- DATE: parsed date equality

F1 scoring for OOLONG-Pairs follows Zhang et al. (2512.24601).
S-NIAH scoring follows RULER's string_match_all (Hsieh et al., 2024).
"""

from __future__ import annotations

import re


def oolong_score(predicted_text: str, ground_truth, answer_type: str) -> float:
    """Score an OOLONG answer based on its answer type.

    Matches the official OOLONG evaluation code (synth_process_response).

    Args:
        predicted_text: Raw model output text.
        ground_truth: Parsed ground truth value from the dataset.
        answer_type: One of ANSWER_TYPE.{NUMERIC, LABEL, COMPARISON, USER, DATE, MONTH_YEAR}.
    """
    parsed = _parse_oolong_output(predicted_text)

    if answer_type == "ANSWER_TYPE.NUMERIC":
        return _score_numeric(parsed, ground_truth)
    if answer_type == "ANSWER_TYPE.COMPARISON":
        return _score_comparison(parsed, ground_truth)
    if answer_type == "ANSWER_TYPE.DATE":
        return _score_date(parsed, ground_truth)
    # LABEL, USER, MONTH_YEAR: exact string match
    return _score_exact_string(parsed, ground_truth)


def _parse_oolong_output(text: str) -> str:
    """Parse the answer from model output.

    Following OOLONG's synth_attempt_answer_parse: split on ":", take the last
    segment after known prefixes (Label:, Answer:, User:, Date:).

    Handles LaTeX formatting that models often produce (e.g. \\boxed{\\text{Answer: 11}}).
    """
    text = text.strip()

    # Strip LaTeX wrappers before parsing
    # Remove \boxed{...} → contents, \text{...} → contents
    text = re.sub(r"\\boxed\{(.+?)\}", r"\1", text)
    text = re.sub(r"\\text\{(.+?)\}", r"\1", text)
    # Remove remaining LaTeX: $, \, leftover braces at boundaries
    text = text.replace("$", "").strip()

    # Look for structured answer patterns
    for prefix in ["Label:", "Answer:", "User:", "Date:", "Month:"]:
        if prefix.lower() in text.lower():
            idx = text.lower().rfind(prefix.lower())
            answer = text[idx + len(prefix):].strip()
            # Take only first line after prefix
            answer = answer.split("\n")[0].strip()
            # Strip common formatting including braces
            answer = answer.strip("*[]'\"` {}")
            return answer

    # Fallback: last line
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    if lines:
        answer = lines[-1].strip("*[]'\"` {}")
        return answer

    return text.strip()


def _score_numeric(parsed: str, gold) -> float:
    """NUMERIC scoring: 0.75^|gold - predicted|."""
    try:
        pred_val = float(parsed.replace(",", ""))
        gold_val = float(gold)
        return 0.75 ** abs(gold_val - pred_val)
    except (ValueError, TypeError):
        return 0.0


def _score_comparison(parsed: str, gold) -> float:
    """COMPARISON scoring: substring match."""
    parsed_lower = parsed.lower().strip()
    gold_str = str(gold).lower().strip()

    if parsed_lower == gold_str:
        return 1.0

    # Partial match: "more common" matches "more common than"
    comparisons = ["more common", "less common", "same frequency"]
    for comp in comparisons:
        if comp in parsed_lower and comp in gold_str:
            return 1.0

    return 0.0


def _score_date(parsed: str, gold) -> float:
    """DATE scoring: parsed date equality."""
    import datetime

    try:
        from dateutil import parser as dateutil_parser
        parsed_date = dateutil_parser.parse(parsed).date()
        if isinstance(gold, datetime.date):
            return 1.0 if parsed_date == gold else 0.0
        gold_date = dateutil_parser.parse(str(gold)).date()
        return 1.0 if parsed_date == gold_date else 0.0
    except Exception:
        # Fallback to string comparison
        return 1.0 if str(parsed).strip() == str(gold).strip() else 0.0


def _score_exact_string(parsed: str, gold) -> float:
    """Exact string match (case-insensitive, stripped)."""
    return 1.0 if str(parsed).strip().lower() == str(gold).strip().lower() else 0.0


# Keep backward-compatible oolong_accuracy for existing code
def oolong_accuracy(predicted: dict[str, float], ground_truth: dict[str, float]) -> float:
    """Legacy OOLONG scoring: 0.75^|y - yhat| for each category, then average.

    Kept for backward compatibility with tests that use the old category-count format.
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
    """F1 score for OOLONG-Pairs: matching user pairs.

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
    """S-NIAH scoring: case-insensitive containment match.

    Follows RULER's string_match_all: checks whether the ground truth string
    appears as a substring within the predicted text.
    """
    return 1.0 if ground_truth.strip().lower() in predicted.strip().lower() else 0.0
