"""Ablation results dashboard.

Loads JSON result files from ablation output directory and displays
a summary ASCII table comparing all conditions with cost/quality tradeoffs.

Usage:
    uv run python -m examples.dashboard [results_dir]
    uv run python -m examples.dashboard ablation_results/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_results(results_dir: Path) -> list[dict]:
    """Load all JSON result files from a directory."""
    all_data = []
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        for entry in data:
            entry["source_file"] = path.name
        all_data.extend(data)
    return all_data


def print_summary_table(data: list[dict]) -> None:
    """Print an overview table of all ablation conditions."""
    # Group by condition
    conditions: dict[str, list[dict]] = {}
    for entry in data:
        cond = entry["condition"]
        conditions.setdefault(cond, []).append(entry)

    # Header
    print(f"\n{'Condition':<30} | {'Runs':>4} | {'Consensus':>9} | {'Confidence':>22} | "
          f"{'Avg Latency':>11} | {'Errors':>6} | {'Avg Div':>8} | {'Avg Jacc':>9}")
    print("-" * 120)

    for cond, entries in conditions.items():
        n = len(entries)
        consensus_count = sum(1 for e in entries if e.get("consensus_reached"))
        consensus_pct = consensus_count / n * 100 if n else 0

        # Confidence distribution
        conf_counts: dict[str, int] = {}
        for e in entries:
            c = e.get("confidence", "?")
            conf_counts[c] = conf_counts.get(c, 0) + 1
        conf_str = ", ".join(f"{k}:{v}" for k, v in sorted(conf_counts.items()))

        avg_latency = sum(e.get("latency_ms", 0) for e in entries) / n if n else 0
        total_errors = sum(e.get("num_errors", 0) for e in entries)

        # Metrics (may not be present)
        divs = [e["metrics"]["avg_reasoning_divergence"] for e in entries if "metrics" in e]
        jaccs = [e["metrics"]["avg_jaccard_distance"] for e in entries if "metrics" in e]
        avg_div = sum(divs) / len(divs) if divs else 0
        avg_jacc = sum(jaccs) / len(jaccs) if jaccs else 0

        print(f"{cond:<30} | {n:>4} | {consensus_pct:>8.0f}% | {conf_str:<22} | "
              f"{avg_latency:>9.0f}ms | {total_errors:>6} | {avg_div:>8.3f} | {avg_jacc:>9.3f}")

    print("-" * 120)


def print_cost_quality_table(data: list[dict]) -> None:
    """Print a cost vs quality comparison."""
    conditions: dict[str, list[dict]] = {}
    for entry in data:
        conditions.setdefault(entry["condition"], []).append(entry)

    print(f"\n{'Condition':<30} | {'Avg Time (s)':>12} | {'Avg Traces':>10} | "
          f"{'Model Div':>9} | {'AFDR Count':>10} | {'Quality Score':>13}")
    print("-" * 100)

    for cond, entries in conditions.items():
        n = len(entries)
        avg_time = sum(e.get("elapsed_seconds", 0) for e in entries) / n if n else 0
        avg_traces = sum(e.get("num_traces", 0) for e in entries) / n if n else 0

        model_divs = [e["metrics"]["model_diversity"] for e in entries if "metrics" in e]
        avg_model_div = sum(model_divs) / len(model_divs) if model_divs else 0

        afdr_counts = [e["metrics"]["afdr_count"] for e in entries if "metrics" in e]
        total_afdr = sum(afdr_counts)

        # Quality score: weighted combination of metrics
        # Higher divergence + higher AFDR = better independence
        quality_scores = []
        for e in entries:
            if "metrics" in e:
                m = e["metrics"]
                score = (
                    m["avg_reasoning_divergence"] * 0.3
                    + m["avg_jaccard_distance"] * 0.2
                    + m["model_diversity"] * 0.3
                    + (1.0 if m["afdr_count"] > 0 else 0.0) * 0.2
                )
                quality_scores.append(score)
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        print(f"{cond:<30} | {avg_time:>10.1f}s | {avg_traces:>10.1f} | "
              f"{avg_model_div:>9.3f} | {total_afdr:>10} | {avg_quality:>13.3f}")

    print("-" * 100)


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ablation_results")

    if not results_dir.exists():
        print(f"No results directory found at {results_dir}")
        print("Run ablation scripts first to generate results.")
        sys.exit(1)

    json_files = list(results_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON result files found in {results_dir}")
        sys.exit(1)

    data = load_results(results_dir)
    print(f"\nLoaded {len(data)} results from {len(json_files)} files in {results_dir}")

    print("\n" + "=" * 120)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 120)
    print_summary_table(data)

    print("\n" + "=" * 100)
    print("COST vs QUALITY TRADEOFFS")
    print("=" * 100)
    print_cost_quality_table(data)

    # Per-problem breakdown
    problems = sorted(set(e["problem"] for e in data))
    print(f"\n{'=' * 80}")
    print("PER-PROBLEM BREAKDOWN")
    print(f"{'=' * 80}")

    for problem in problems:
        print(f"\n--- {problem} ---")
        entries = [e for e in data if e["problem"] == problem]
        for e in entries:
            conf = e.get("confidence", "?")
            consensus = "Y" if e.get("consensus_reached") else "N"
            latency = e.get("latency_ms", 0)
            print(f"  {e['condition']:<30} | {conf:<12} | consensus={consensus} | {latency:.0f}ms")


if __name__ == "__main__":
    main()
