"""Tests for benchmark infrastructure (scoring, datasets, runner)."""

from __future__ import annotations

import datetime
import json
import tempfile
from pathlib import Path

import pytest

from rde.benchmarks.datasets import (
    OolongPairsLoader,
    SNIAHLoader,
    _at_least,
    _build_query_text,
    _check_pair,
    _exactly,
    _parse_oolong_answer,
    OOLONG_PAIRS_QUERIES,
)
from rde.benchmarks.runner import BenchmarkResult, BenchmarkRunner, RLM_BASELINES, TaskResult
from rde.benchmarks.scoring import (
    _parse_oolong_output,
    exact_match,
    f1_pairs,
    oolong_accuracy,
    oolong_score,
)


# --- OOLONG scoring tests (per answer type) ---


def test_oolong_score_numeric_exact():
    """NUMERIC answer type: exact match scores 1.0."""
    assert oolong_score("Answer: 7", 7, "ANSWER_TYPE.NUMERIC") == pytest.approx(1.0)


def test_oolong_score_numeric_off_by_one():
    """NUMERIC answer type: off by 1 scores 0.75."""
    assert oolong_score("Answer: 8", 7, "ANSWER_TYPE.NUMERIC") == pytest.approx(0.75)


def test_oolong_score_numeric_parse_fail():
    """NUMERIC answer type: unparseable answer scores 0.0."""
    assert oolong_score("I don't know", 7, "ANSWER_TYPE.NUMERIC") == 0.0


def test_oolong_score_label_exact():
    """LABEL answer type: exact match."""
    assert oolong_score("Label: incorrect", "incorrect", "ANSWER_TYPE.LABEL") == 1.0


def test_oolong_score_label_wrong():
    """LABEL answer type: wrong label scores 0.0."""
    assert oolong_score("Label: correct", "incorrect", "ANSWER_TYPE.LABEL") == 0.0


def test_oolong_score_comparison():
    """COMPARISON answer type: substring match."""
    assert oolong_score("Answer: more common", "more common than", "ANSWER_TYPE.COMPARISON") == 1.0
    assert oolong_score("Answer: less common", "less common than", "ANSWER_TYPE.COMPARISON") == 1.0
    assert oolong_score("Answer: more common", "less common than", "ANSWER_TYPE.COMPARISON") == 0.0


def test_oolong_score_user():
    """USER answer type: exact string match on ID."""
    assert oolong_score("User: 44106", 44106, "ANSWER_TYPE.USER") == 1.0
    assert oolong_score("User: 99999", 44106, "ANSWER_TYPE.USER") == 0.0


def test_oolong_score_month_year():
    """MONTH_YEAR answer type: exact string match."""
    assert oolong_score("Answer: October 2022", "October 2022", "ANSWER_TYPE.MONTH_YEAR") == 1.0


# --- Legacy oolong_accuracy tests ---


def test_oolong_accuracy_perfect():
    """Perfect prediction scores 1.0."""
    gt = {"ABBR": 10, "DESC": 20, "ENTY": 15, "HUM": 5, "LOC": 30, "NUM": 20}
    assert oolong_accuracy(gt, gt) == pytest.approx(1.0)


def test_oolong_accuracy_off_by_one():
    """Off-by-one predictions score 0.75 per category."""
    gt = {"ABBR": 10, "DESC": 20}
    pred = {"ABBR": 11, "DESC": 21}
    assert oolong_accuracy(pred, gt) == pytest.approx(0.75)


def test_oolong_accuracy_empty():
    """Empty ground truth returns 0."""
    assert oolong_accuracy({"A": 1}, {}) == 0.0


def test_oolong_accuracy_missing_category():
    """Missing predicted category counts as 0."""
    gt = {"ABBR": 5, "DESC": 10}
    pred = {"ABBR": 5}  # DESC missing -> predicted as 0
    score = oolong_accuracy(pred, gt)
    assert 0.5 < score < 0.6  # Average of 1.0 and 0.75^10


# --- F1 pairs tests ---


def test_f1_pairs_perfect():
    """Perfect prediction scores 1.0."""
    pairs = {("a", "b"), ("c", "d")}
    assert f1_pairs(pairs, pairs) == pytest.approx(1.0)


def test_f1_pairs_partial():
    """Partial match gives intermediate F1."""
    pred = {("a", "b"), ("c", "d")}
    truth = {("a", "b"), ("e", "f")}
    assert f1_pairs(pred, truth) == pytest.approx(0.5)


def test_f1_pairs_order_invariant():
    """Pair order doesn't matter: (a,b) == (b,a)."""
    pred = {("b", "a")}
    truth = {("a", "b")}
    assert f1_pairs(pred, truth) == pytest.approx(1.0)


def test_f1_pairs_empty():
    """Both empty returns 1.0; one empty returns 0.0."""
    assert f1_pairs(set(), set()) == pytest.approx(1.0)
    assert f1_pairs(set(), {("a", "b")}) == pytest.approx(0.0)
    assert f1_pairs({("a", "b")}, set()) == pytest.approx(0.0)


# --- Exact match tests ---


def test_exact_match_case_insensitive():
    """Exact match is case-insensitive containment."""
    assert exact_match("Hello", "hello") == 1.0
    assert exact_match("Hello World", "hello") == 1.0
    assert exact_match("Hello", "world") == 0.0


def test_exact_match_whitespace():
    """Exact match strips whitespace."""
    assert exact_match("  hello  ", "hello") == 1.0


def test_exact_match_containment():
    """Exact match supports containment (LLMs wrap answers)."""
    assert exact_match("The answer is 4829371.", "4829371") == 1.0


# --- OOLONG answer parser tests ---


def test_parse_oolong_answer_numeric():
    """Parse numeric answer from oolong-synth format."""
    assert _parse_oolong_answer("[7]") == 7


def test_parse_oolong_answer_string():
    """Parse string answer from oolong-synth format."""
    assert _parse_oolong_answer("['incorrect']") == "incorrect"


def test_parse_oolong_answer_comparison():
    """Parse comparison answer."""
    assert _parse_oolong_answer("['more common than']") == "more common than"


# --- OOLONG output parser (LaTeX handling) tests ---


def test_parse_oolong_output_latex_boxed_text():
    """Parse answer from LaTeX \\boxed{\\text{Answer: 11}}."""
    assert _parse_oolong_output(r"$\boxed{\text{Answer: 11}}$") == "11"


def test_parse_oolong_output_latex_label():
    """Parse label from LaTeX \\text{Label: positive}."""
    assert _parse_oolong_output(r"$\boxed{\text{Label: positive}}$") == "positive"


def test_parse_oolong_output_plain_prefix():
    """Parse plain Answer: prefix (no LaTeX)."""
    assert _parse_oolong_output("Answer: 42") == "42"


def test_parse_oolong_output_trailing_braces():
    """Trailing braces from LaTeX are stripped."""
    assert _parse_oolong_output("Answer: hello}}") == "hello"


def test_parse_oolong_output_multiline_last_line():
    """Fallback to last line when no prefix found."""
    assert _parse_oolong_output("Some reasoning\nThe answer\n42") == "42"


# --- OOLONG-Pairs constraint tests ---


def test_pairs_symmetric_condition():
    """Symmetric condition checks both users have matching labels."""
    query = OOLONG_PAIRS_QUERIES[0]  # Task 1: numeric value or location
    user_a = [{"label": "numeric value", "date": datetime.date(2023, 3, 1)}]
    user_b = [{"label": "location", "date": datetime.date(2023, 3, 1)}]
    user_c = [{"label": "entity", "date": datetime.date(2023, 3, 1)}]

    assert _check_pair(query, user_a, user_b) is True  # Both match
    assert _check_pair(query, user_a, user_c) is False  # user_c has no match


def test_pairs_temporal_condition():
    """Temporal condition enforces date constraints."""
    query = OOLONG_PAIRS_QUERIES[3]  # Task 4: human being or location, human being after Jan 6 2023
    user_a = [
        {"label": "human being", "date": datetime.date(2023, 6, 1)},  # After cutoff
    ]
    user_b = [
        {"label": "location", "date": datetime.date(2022, 11, 1)},
    ]
    user_c = [
        {"label": "human being", "date": datetime.date(2022, 11, 1)},  # Before cutoff
    ]

    assert _check_pair(query, user_a, user_b) is True  # Both valid
    assert _check_pair(query, user_a, user_c) is False  # user_c fails temporal


def test_pairs_asymmetric_condition():
    """Asymmetric condition checks different conditions per user."""
    query = OOLONG_PAIRS_QUERIES[10]  # Task 11: one has entity+abbreviation, other has exactly 1 entity
    user_a = [
        {"label": "entity", "date": datetime.date(2023, 1, 1)},
        {"label": "abbreviation", "date": datetime.date(2023, 1, 1)},
    ]
    user_b = [
        {"label": "entity", "date": datetime.date(2023, 1, 1)},
    ]

    assert _check_pair(query, user_a, user_b) is True  # a=cond_a, b=cond_b
    assert _check_pair(query, user_b, user_a) is True  # Reverse order also works


def test_at_least_exactly_helpers():
    """Test _at_least and _exactly helper functions."""
    entries = [
        {"label": "entity"},
        {"label": "entity"},
        {"label": "location"},
    ]
    assert _at_least(entries, "entity", 1) is True
    assert _at_least(entries, "entity", 2) is True
    assert _at_least(entries, "entity", 3) is False
    assert _exactly(entries, "entity", 2) is True
    assert _exactly(entries, "location", 1) is True
    assert _exactly(entries, "location", 2) is False


def test_build_query_text():
    """Query text includes constraint and label instruction."""
    query = OOLONG_PAIRS_QUERIES[0]
    text = _build_query_text(query)
    assert "list all pairs of user IDs" in text
    assert "numeric value or location" in text
    assert "description and abstract concept" in text  # From label instruction


# --- OOLONG-Pairs loader tests ---


def test_oolong_pairs_loader_generates_tasks():
    """OolongPairsLoader produces pairwise matching tasks."""
    dataset = OolongPairsLoader.load(
        context_size_tokens=1_000,
        num_tasks=2,
    )
    assert dataset.num_tasks == 2
    assert dataset.name == "oolong_pairs"

    for task in dataset.tasks:
        assert task.scoring_fn == "f1_pairs"
        assert isinstance(task.ground_truth, set)
        assert "num_users" in task.metadata
        assert "num_gt_pairs" in task.metadata


def test_oolong_pairs_reproducible():
    """Same seed produces same tasks."""
    d1 = OolongPairsLoader.load(context_size_tokens=500, num_tasks=2, seed=42)
    d2 = OolongPairsLoader.load(context_size_tokens=500, num_tasks=2, seed=42)
    assert d1.tasks[0].task_id == d2.tasks[0].task_id
    assert d1.tasks[0].ground_truth == d2.tasks[0].ground_truth


# --- S-NIAH loader tests ---


def test_sniah_loader_generates_tasks():
    """SNIAHLoader produces RULER-style needle-in-haystack tasks."""
    dataset = SNIAHLoader.load(context_sizes=[1_000], tasks_per_size=3)
    assert dataset.num_tasks == 3
    assert dataset.name == "sniah"

    for task in dataset.tasks:
        assert task.scoring_fn == "exact_match"
        assert isinstance(task.ground_truth, str)
        assert len(task.context) > 100
        # Needle should be present in the context
        assert task.ground_truth in task.context


def test_sniah_ruler_format():
    """S-NIAH tasks follow RULER template format."""
    dataset = SNIAHLoader.load(context_sizes=[1_000], tasks_per_size=3)

    for task in dataset.tasks:
        # Context should have RULER preamble
        assert "special magic" in task.context
        assert "memorize" in task.context
        # Query should ask for the specific key
        assert "What is the special magic" in task.query
        # Metadata should have subtask info
        assert "subtask" in task.metadata
        assert task.metadata["value_type"] in ("numbers", "uuids")


def test_sniah_reproducible():
    """Same seed produces same S-NIAH tasks."""
    d1 = SNIAHLoader.load(context_sizes=[1_000], tasks_per_size=2, seed=42)
    d2 = SNIAHLoader.load(context_sizes=[1_000], tasks_per_size=2, seed=42)
    assert d1.tasks[0].task_id == d2.tasks[0].task_id
    assert d1.tasks[0].ground_truth == d2.tasks[0].ground_truth


# --- Runner tests ---


def test_rlm_baselines_populated():
    """RLM_BASELINES contains expected benchmarks and methods."""
    assert "oolong" in RLM_BASELINES
    assert "oolong_pairs" in RLM_BASELINES
    assert "sniah" in RLM_BASELINES
    assert "GPT-5 Base" in RLM_BASELINES["oolong"]
    assert "RLM (GPT-5)" in RLM_BASELINES["oolong"]


def test_comparison_table_format():
    """comparison_table() produces valid markdown."""
    results = {
        "oolong": BenchmarkResult(
            benchmark_name="oolong",
            config_name="Test Config",
            aggregate_score=55.5,
            aggregate_metric="accuracy",
        ),
    }
    table = BenchmarkRunner.comparison_table(results)
    assert "| Method |" in table
    assert "Test Config" in table
    assert "55.5" in table
    assert "RLM" in table  # RLM baselines included


def test_comparison_table_no_rlm():
    """comparison_table() can exclude RLM baselines."""
    results = {
        "oolong": BenchmarkResult(
            benchmark_name="oolong",
            config_name="Test",
            aggregate_score=50.0,
        ),
    }
    table = BenchmarkRunner.comparison_table(results, include_rlm=False)
    assert "RLM" not in table


def test_save_results_json():
    """save_results() writes valid JSON."""
    results = {
        "oolong": BenchmarkResult(
            benchmark_name="oolong",
            config_name="Test",
            aggregate_score=45.0,
            aggregate_metric="accuracy",
            task_results=[
                TaskResult(task_id="t1", predicted="x", ground_truth="y", score=0.5),
            ],
        ),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "results.json"
        BenchmarkRunner.save_results(results, path)

        assert path.exists()
        data = json.loads(path.read_text())
        assert "oolong" in data
        assert data["oolong"]["aggregate_score"] == 45.0
        assert len(data["oolong"]["task_scores"]) == 1


def test_parse_category_counts():
    """_parse_category_counts() extracts JSON from text."""
    text = 'The distribution is {"ABBR": 5, "DESC": 10}'
    result = BenchmarkRunner._parse_category_counts(text)
    assert result == {"ABBR": 5, "DESC": 10}


def test_parse_category_counts_no_json():
    """_parse_category_counts() returns empty dict for non-JSON."""
    assert BenchmarkRunner._parse_category_counts("no json here") == {}


def test_parse_user_pairs_tuple_format():
    """_parse_user_pairs() extracts (id1, id2) formatted pairs."""
    text = "(12345, 67890)\n(11111, 22222)"
    result = BenchmarkRunner._parse_user_pairs(text)
    assert ("12345", "67890") in result
    assert ("11111", "22222") in result


def test_parse_user_pairs_json_format():
    """_parse_user_pairs() handles JSON array format."""
    text = '[["12345", "67890"], ["11111", "22222"]]'
    result = BenchmarkRunner._parse_user_pairs(text)
    assert ("12345", "67890") in result
    assert ("11111", "22222") in result


def test_parse_user_pairs_no_data():
    """_parse_user_pairs() returns empty set for no pairs."""
    assert BenchmarkRunner._parse_user_pairs("no pairs here") == set()


def test_parse_pairs():
    """_parse_pairs() extracts pair lists from text (legacy)."""
    text = 'Found pairs: [["doc_1", "doc_3"], ["doc_2", "doc_5"]]'
    result = BenchmarkRunner._parse_pairs(text)
    assert ("doc_1", "doc_3") in result
    assert ("doc_2", "doc_5") in result


def test_parse_pairs_no_json():
    """_parse_pairs() returns empty set for non-JSON."""
    assert BenchmarkRunner._parse_pairs("no pairs") == set()
