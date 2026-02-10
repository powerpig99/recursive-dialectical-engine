"""Tests for benchmark infrastructure (scoring, datasets, runner)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from rde.benchmarks.datasets import BenchmarkDataset, OolongLoader, SNIAHLoader
from rde.benchmarks.runner import BenchmarkResult, BenchmarkRunner, RLM_BASELINES, TaskResult
from rde.benchmarks.scoring import exact_match, f1_pairs, oolong_accuracy


# --- Scoring tests ---


def test_oolong_accuracy_perfect():
    """Perfect prediction scores 1.0."""
    gt = {"ABBR": 10, "DESC": 20, "ENTY": 15, "HUM": 5, "LOC": 30, "NUM": 20}
    assert oolong_accuracy(gt, gt) == pytest.approx(1.0)


def test_oolong_accuracy_off_by_one():
    """Off-by-one predictions score 0.75 per category."""
    gt = {"ABBR": 10, "DESC": 20}
    pred = {"ABBR": 11, "DESC": 21}
    # Each category: 0.75^1 = 0.75
    assert oolong_accuracy(pred, gt) == pytest.approx(0.75)


def test_oolong_accuracy_empty():
    """Empty ground truth returns 0."""
    assert oolong_accuracy({"A": 1}, {}) == 0.0


def test_oolong_accuracy_missing_category():
    """Missing predicted category counts as 0."""
    gt = {"ABBR": 5, "DESC": 10}
    pred = {"ABBR": 5}  # DESC missing → predicted as 0
    # ABBR: 0.75^0 = 1.0, DESC: 0.75^10 ≈ 0.056
    score = oolong_accuracy(pred, gt)
    assert 0.5 < score < 0.6  # Average of 1.0 and 0.056


def test_f1_pairs_perfect():
    """Perfect prediction scores 1.0."""
    pairs = {("a", "b"), ("c", "d")}
    assert f1_pairs(pairs, pairs) == pytest.approx(1.0)


def test_f1_pairs_partial():
    """Partial match gives intermediate F1."""
    pred = {("a", "b"), ("c", "d")}
    truth = {("a", "b"), ("e", "f")}
    # TP=1, precision=1/2, recall=1/2, F1=0.5
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


def test_exact_match_case_insensitive():
    """Exact match is case-insensitive."""
    assert exact_match("Hello", "hello") == 1.0
    assert exact_match("Hello", "world") == 0.0


def test_exact_match_whitespace():
    """Exact match strips whitespace."""
    assert exact_match("  hello  ", "hello") == 1.0


# --- Dataset tests ---


def test_oolong_loader_generates_tasks():
    """OolongLoader produces the requested number of tasks."""
    dataset = OolongLoader.load_trec_coarse(
        context_size_tokens=1_000,  # Small for testing
        num_tasks=3,
    )
    assert isinstance(dataset, BenchmarkDataset)
    assert dataset.num_tasks == 3
    assert dataset.name == "oolong"

    for task in dataset.tasks:
        assert task.scoring_fn == "oolong_accuracy"
        assert isinstance(task.ground_truth, dict)
        assert len(task.context) > 0


def test_oolong_pairs_loader_generates_tasks():
    """OolongLoader.load_oolong_pairs produces pairwise matching tasks."""
    dataset = OolongLoader.load_oolong_pairs(
        context_size_tokens=1_000,
        num_tasks=2,
    )
    assert dataset.num_tasks == 2
    assert dataset.name == "oolong_pairs"

    for task in dataset.tasks:
        assert task.scoring_fn == "f1_pairs"
        assert isinstance(task.ground_truth, set)


def test_sniah_loader_generates_tasks():
    """SNIAHLoader produces needle-in-haystack tasks."""
    dataset = SNIAHLoader.load(context_sizes=[1_000], tasks_per_size=3)
    assert dataset.num_tasks == 3
    assert dataset.name == "sniah"

    for task in dataset.tasks:
        assert task.scoring_fn == "exact_match"
        assert isinstance(task.ground_truth, str)
        # The needle should be present in the context
        assert len(task.context) > 100


def test_dataset_reproducible():
    """Same seed produces same tasks."""
    d1 = OolongLoader.load_trec_coarse(context_size_tokens=500, num_tasks=2, seed=42)
    d2 = OolongLoader.load_trec_coarse(context_size_tokens=500, num_tasks=2, seed=42)
    assert d1.tasks[0].task_id == d2.tasks[0].task_id
    assert d1.tasks[0].ground_truth == d2.tasks[0].ground_truth


# --- Runner tests ---


def test_rlm_baselines_populated():
    """RLM_BASELINES contains expected benchmarks and methods."""
    assert "oolong" in RLM_BASELINES
    assert "oolong_pairs" in RLM_BASELINES
    assert "sniah" in RLM_BASELINES
    assert "GPT-4o Base" in RLM_BASELINES["oolong"]
    assert "RLM (GPT-4o)" in RLM_BASELINES["oolong"]


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


def test_parse_pairs():
    """_parse_pairs() extracts pair lists from text."""
    text = 'Found pairs: [["doc_1", "doc_3"], ["doc_2", "doc_5"]]'
    result = BenchmarkRunner._parse_pairs(text)
    assert ("doc_1", "doc_3") in result
    assert ("doc_2", "doc_5") in result


def test_parse_pairs_no_json():
    """_parse_pairs() returns empty set for non-JSON."""
    assert BenchmarkRunner._parse_pairs("no pairs") == set()
