#!/usr/bin/env bash
# Run all local benchmark configs overnight.
# Outputs to results/local/ with timestamps.
#
# Usage: ./scripts/run_overnight_local.sh
set -uo pipefail

export LOCAL_OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export LOCAL_OPENAI_MODEL="Qwen/Qwen3-8B-MLX-4bit"

RUN="uv run python -m examples.benchmarks.run_local_benchmarks"
LOG="results/local/overnight_$(date +%Y%m%d_%H%M%S).log"
SUMMARY_FILE="results/local/overnight_summary.txt"
mkdir -p results/local

echo "=== Overnight Local Benchmark Run ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Clear summary file
> "$SUMMARY_FILE"

run_bench() {
    local config="$1"
    local bench="$2"
    local ctx="$3"
    local tasks="$4"
    local label="${config}/${bench}@${ctx}"

    echo "" | tee -a "$LOG"
    echo ">>> [$label] Starting at $(date)" | tee -a "$LOG"

    output=$($RUN --config "$config" --benchmark "$bench" --context-size "$ctx" --max-tasks "$tasks" 2>&1)
    exit_code=$?

    echo "$output" >> "$LOG"

    # Extract score from output
    score=$(echo "$output" | grep -oE '[0-9]+\.[0-9]+% accuracy|[0-9]+\.[0-9]+% F1' | head -1)
    if [ -z "$score" ]; then
        score="FAILED (exit $exit_code)"
    fi

    echo "<<< [$label] Done at $(date): $score" | tee -a "$LOG"
    printf "%-50s %s\n" "$label" "$score" >> "$SUMMARY_FILE"
}

# ============================================================
# Phase 1: Baselines (fast — ~1.5 hours)
# ============================================================
echo "=== Phase 1: Vanilla baselines (50 tasks) ===" | tee -a "$LOG"

run_bench local_vanilla oolong 1024 50
run_bench local_vanilla sniah 1024 10
run_bench local_vanilla oolong_pairs 4096 10

# ============================================================
# Phase 2: REPL (RLM equivalent — ~2-3 hours)
# ============================================================
echo "" | tee -a "$LOG"
echo "=== Phase 2: REPL (single model + REPL) ===" | tee -a "$LOG"

run_bench local_repl oolong 1024 50
run_bench local_repl sniah 1024 10
run_bench local_repl oolong_pairs 4096 10

# ============================================================
# Phase 3: Dialectical same-model (key comparison — ~3 hours)
# ============================================================
echo "" | tee -a "$LOG"
echo "=== Phase 3: Dialectical 3x same model ===" | tee -a "$LOG"

run_bench local_dialectical_same oolong 1024 50
run_bench local_dialectical_same sniah 1024 10
run_bench local_dialectical_same oolong_pairs 4096 10

# ============================================================
# Phase 4: Dialectical + REPL (full RDE — ~4-5 hours)
# ============================================================
echo "" | tee -a "$LOG"
echo "=== Phase 4: Dialectical 3x + REPL ===" | tee -a "$LOG"

run_bench local_dialectical_repl_same oolong 1024 30
run_bench local_dialectical_repl_same sniah 1024 10
run_bench local_dialectical_repl_same oolong_pairs 4096 10

# ============================================================
# Phase 5: Dialectical 2-iteration (shadow reframing — ~4-5 hours)
# ============================================================
echo "" | tee -a "$LOG"
echo "=== Phase 5: Dialectical 3x, 2 iterations ===" | tee -a "$LOG"

run_bench local_dialectical_2iter oolong 1024 30
run_bench local_dialectical_2iter sniah 1024 10
run_bench local_dialectical_2iter oolong_pairs 4096 10

# ============================================================
# Final summary
# ============================================================
echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "FINAL SUMMARY" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

sort "$SUMMARY_FILE" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Full log: $LOG" | tee -a "$LOG"
echo "JSON results: results/local/*.json" | tee -a "$LOG"

# Run analysis
echo "" | tee -a "$LOG"
echo "=== Analysis ===" | tee -a "$LOG"
uv run python -m examples.benchmarks.analyze_results --dir results/local 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== DONE ===" | tee -a "$LOG"
