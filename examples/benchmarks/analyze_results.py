"""Analyze local benchmark results and produce comparison tables.

Loads all result JSONs from results/local/ and produces:
- Comparison tables (markdown)
- Improvement %s vs baseline
- Independence metric summaries
- Per-benchmark failure analysis

Usage:
    python -m examples.benchmarks.analyze_results
    python -m examples.benchmarks.analyze_results --dir results/local
    python -m examples.benchmarks.analyze_results --model Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_results(result_dir: Path, model_filter: str | None = None) -> list[dict]:
    """Load all result JSON files from directory."""
    results = []
    for path in sorted(result_dir.glob("*.json")):
        if model_filter and model_filter.lower() not in path.name.lower():
            continue
        try:
            data = json.loads(path.read_text())
            results.append({"file": path.name, "data": data})
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Skipping {path.name}: {e}", file=sys.stderr)
    return results


def extract_scores(data: dict) -> dict[str, dict[str, float]]:
    """Extract config/benchmark -> score mapping from a result file."""
    scores: dict[str, dict[str, float]] = {}
    for key, entry in data.items():
        config_name = entry.get("config_name", key)
        benchmark = entry.get("benchmark_name", key.split("/")[-1] if "/" in key else key)
        score = entry.get("aggregate_score", 0.0)
        if config_name not in scores:
            scores[config_name] = {}
        scores[config_name][benchmark] = score
    return scores


def extract_independence(data: dict) -> dict[str, dict]:
    """Extract independence metrics from results."""
    metrics: dict[str, dict] = {}
    for key, entry in data.items():
        if "independence" in entry:
            config_name = entry.get("config_name", key)
            metrics[config_name] = entry["independence"]
    return metrics


def print_comparison_table(all_scores: dict[str, dict[str, float]]) -> None:
    """Print a markdown comparison table."""
    benchmarks = set()
    for scores in all_scores.values():
        benchmarks.update(scores.keys())
    benchmarks = sorted(benchmarks)

    if not benchmarks:
        print("No benchmark results found.")
        return

    # Header
    bench_headers = " | ".join(f"{b}" for b in benchmarks)
    print(f"\n| Config | {bench_headers} |")
    print("|" + "-" * 40 + "|" + "|".join("-" * 14 for _ in benchmarks) + "|")

    # Rows sorted by first benchmark score
    sorted_configs = sorted(
        all_scores.items(),
        key=lambda x: list(x[1].values())[0] if x[1] else 0,
        reverse=True,
    )

    for config_name, scores in sorted_configs:
        cells = []
        for b in benchmarks:
            s = scores.get(b)
            cells.append(f"{s:.1f}%" if s is not None else "-")
        print(f"| {config_name:38s} | " + " | ".join(f"{c:>12s}" for c in cells) + " |")


def print_ablation_analysis(all_scores: dict[str, dict[str, float]]) -> None:
    """Print ablation comparison (vanilla vs repl vs dialectical)."""
    print("\n## Ablation Analysis\n")

    # Find vanilla baseline
    vanilla_scores: dict[str, float] = {}
    for config_name, scores in all_scores.items():
        if "vanilla" in config_name.lower():
            vanilla_scores = scores
            break

    if not vanilla_scores:
        print("No vanilla baseline found for comparison.")
        return

    for config_name, scores in all_scores.items():
        if "vanilla" in config_name.lower():
            continue
        improvements = []
        for bench, score in scores.items():
            base = vanilla_scores.get(bench)
            if base is not None and base > 0:
                delta = score - base
                pct = (delta / base) * 100
                improvements.append(f"  {bench}: {score:.1f}% (baseline {base:.1f}%, delta {delta:+.1f}pp, {pct:+.1f}%)")
            elif base is not None:
                improvements.append(f"  {bench}: {score:.1f}% (baseline {base:.1f}%)")

        if improvements:
            print(f"**{config_name}** vs vanilla:")
            for line in improvements:
                print(line)
            print()


def print_independence_summary(all_independence: dict[str, dict]) -> None:
    """Print independence metrics summary."""
    if not all_independence:
        return

    print("\n## Independence Metrics\n")
    print("| Config | Agreement | Diversity | Divergence | Jaccard | AFDR |")
    print("|--------|-----------|-----------|------------|---------|------|")

    for config_name, metrics in all_independence.items():
        ca = metrics.get("conclusion_agreement", 0)
        mds = metrics.get("model_diversity_score", 0)
        ard = metrics.get("avg_reasoning_divergence", 0)
        ajd = metrics.get("avg_jaccard_distance", 0)
        afdr = metrics.get("agreement_for_different_reasons_count", 0)
        print(f"| {config_name:30s} | {ca:.3f}     | {mds:.3f}     | {ard:.3f}      | {ajd:.3f}   | {afdr}    |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze local benchmark results")
    parser.add_argument(
        "--dir", type=str, default="results/local",
        help="Results directory (default: results/local)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Filter by model name substring",
    )
    args = parser.parse_args()

    result_dir = Path(args.dir)
    if not result_dir.exists():
        print(f"No results directory found at {result_dir}")
        sys.exit(1)

    results = load_results(result_dir, args.model)
    if not results:
        print(f"No result files found in {result_dir}")
        sys.exit(1)

    print("# Local Benchmark Results\n")
    print(f"Files: {len(results)} from {result_dir}\n")

    # Aggregate scores across all files
    all_scores: dict[str, dict[str, float]] = {}
    all_independence: dict[str, dict] = {}

    for entry in results:
        print(f"### {entry['file']}")
        scores = extract_scores(entry["data"])
        independence = extract_independence(entry["data"])

        all_scores.update(scores)
        all_independence.update(independence)

        # Per-file table
        print_comparison_table(scores)
        print()

    # Overall comparison
    if len(results) > 1:
        print("\n## Overall Comparison (all files)")
        print_comparison_table(all_scores)

    # Ablation analysis
    print_ablation_analysis(all_scores)

    # Independence
    print_independence_summary(all_independence)


if __name__ == "__main__":
    main()
