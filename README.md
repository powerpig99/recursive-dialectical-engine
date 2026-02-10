# Recursive Dialectical Engine (RDE)

Multi-model dialectical reasoning with recursive arbitration.

RDE composes multiple LLM traces into a single resolution through causal arbitration. Each trace operates independently — different model, different role, different perspective on the same externalized context. Where they disagree, a recursive arbiter resolves through causal necessity, not voting. What the resolution necessarily obscures is reported as shadows.

## Core Claim

A single LLM call collapses the problem's dimensionality into one projection. RDE preserves orthogonality by running independent traces across models and composing their outputs through an arbiter that demands causal chains, not majority agreement.

## Installation

```bash
# Clone and install with uv
git clone <repo-url>
cd recursive-dialectical-engine
uv sync

# Install with cloud providers (Anthropic, OpenAI, Google)
uv sync --extra cloud

# Install with local model support (MLX)
uv sync --extra local

# Install dev dependencies
uv sync --extra dev
```

### API Keys

Set the API keys for providers you want to use:

```bash
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
export GOOGLE_API_KEY="..."
export XAI_API_KEY="..."          # optional: xAI/Grok
export KIMI_API_KEY="..."         # optional: Moonshot/Kimi
export HUGGINGFACE_API_KEY="..."  # optional: HuggingFace Inference
```

RDE auto-discovers available providers based on which keys are set. At least one key is required.

### Local Inference (vLLM-mlx)

For local model inference on Apple Silicon via [vLLM-mlx](https://github.com/waybarrios/vllm-mlx):

```bash
# Start the vLLM-mlx server (defaults to Qwen3-8B-MLX-8bit)
./scripts/run_vllm_mlx.sh

# Or with a custom model
MODEL_NAME=Qwen/Qwen3-4B-MLX-4bit ./scripts/run_vllm_mlx.sh

# Tell RDE to use the local server
export LOCAL_OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export LOCAL_OPENAI_MODEL=default
```

Also works with LM Studio, Ollama, or any OpenAI-compatible endpoint. Use `model_preference="local:model-name"` in trace configs to route specific traces to the local server.

## Quick Start

### CLI

```bash
# Basic usage
uv run rde "Should you switch doors in the Monty Hall problem?"

# Multi-iteration with shadow reframing
uv run rde "Is P equal to NP?" --max-iterations 3

# Custom models
uv run rde "What is consciousness?" --trace-models "claude-sonnet-4-5-20250929,gemini-2.5-pro,grok-3"

# With independence metrics
uv run rde "Is free will compatible with determinism?" --metrics

# JSON output
uv run rde "What caused the 2008 financial crisis?" --json

# Save call tree for visualization
uv run rde "Explain dark matter" --trace-log call_tree.json

# Skip orchestrator (use default Believer/Logician/Contrarian)
uv run rde "Is the trolley problem resolvable?" --no-orchestrator

# Read prompt from stdin
echo "Analyze Simpson's paradox" | uv run rde --stdin
```

### Python API

```python
import asyncio
from rde.engine import DialecticalEngine
from rde.models import ModelConfig, RecursionBudget

async def main():
    config = ModelConfig(
        trace_models=["claude-sonnet-4-5-20250929", "gemini-2.5-pro", "grok-3"],
        arbiter_temperature=0.1,       # low temp for deterministic arbitration
        orchestrator_temperature=0.3,   # moderate temp for trace design
    )

    async with DialecticalEngine(config) as engine:
        result = await engine.run(
            prompt="Should you switch doors in the Monty Hall problem?",
            max_iterations=2,
            budget=RecursionBudget(max_depth=3, max_total_calls=20),
        )

    print(f"Resolution: {result.resolution}")
    print(f"Confidence: {result.confidence.value}")
    print(f"Shadows: {result.shadows}")
    print(f"Consensus: {result.consensus_reached}")

asyncio.run(main())
```

## Architecture

```
User Prompt
     │
     ▼
┌─────────────────────┐
│ Context Environment  │  ← Externalized prompt (never passed as raw LM context)
│ peek / search /      │     Traces interact with it programmatically
│ partition / sub_lm   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    Orchestrator      │  ← Designs trace configs via LLM analysis
│ (problem → traces)   │     Determines roles, perspectives, context strategies
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│          Parallel Trace Execution        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ Trace A  │ │ Trace B  │ │ Trace C  │ │  ← Independent: different model,
│  │ Believer │ │ Logician │ │Contrarian│ │     different role, different
│  │ Claude   │ │ Gemini   │ │  Grok    │ │     perspective on same context
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ │
└───────┼────────────┼────────────┼────────┘
        │            │            │
        ▼            ▼            ▼
┌─────────────────────┐
│    Normalizer        │  ← Extracts: conclusion, reasoning chain,
│                      │     confidence, evidence cited
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Consensus Check     │  ← If all traces agree → done (necessary confidence)
└────────┬────────────┘
         │ (disagreement)
         ▼
┌─────────────────────┐
│   Recursive Arbiter  │  ← Resolves via causal necessity, not voting
│   causal_chain       │     Reports shadows (what resolution obscures)
│   shadows            │     Can spawn sub-arbitration on interference
│   confidence         │     dimensions (recursive)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Convergence Check   │  ← Stop if: necessary confidence, budget exhausted,
│                      │     or shadows repeating
└────────┬────────────┘
         │ (novel shadows remain)
         ▼
    Reframe & Iterate   ← Orchestrator designs new traces targeting shadows
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `DialecticalEngine` | `rde/engine.py` | Main orchestration loop |
| `ContextEnvironment` | `rde/environment.py` | Externalized prompt REPL |
| `Orchestrator` | `rde/orchestrator.py` | LLM-driven trace design |
| `TraceExecutor` | `rde/trace.py` | Executes individual traces |
| `TraceNormalizer` | `rde/normalizer.py` | Cross-model output normalization |
| `Arbiter` | `rde/arbiter.py` | Causal disagreement resolution |
| `ModelRouter` | `rde/providers/router.py` | Multi-provider model routing |

### Providers

RDE supports 7 model providers:

| Provider | Models | Key |
|----------|--------|-----|
| Anthropic | Claude Opus, Sonnet, Haiku | `ANTHROPIC_API_KEY` |
| OpenAI | GPT-5, GPT-5-mini | `OPENAI_API_KEY` |
| Google | Gemini 2.5 Pro/Flash | `GOOGLE_API_KEY` |
| xAI | Grok-3 | `XAI_API_KEY` |
| Moonshot | Kimi models | `KIMI_API_KEY` |
| HuggingFace | Inference API models | `HUGGINGFACE_API_KEY` |
| Local (vLLM-mlx) | Any OpenAI-compatible local server | `LOCAL_OPENAI_BASE_URL` |
| MLX (deprecated) | Direct in-process MLX | (local, no key) |

## Configuration

```python
from rde.models import ModelConfig, RecursionBudget

config = ModelConfig(
    # Which models to use
    orchestrator_model="claude-opus-4-6",
    arbiter_model="claude-opus-4-6",
    trace_models=["claude-sonnet-4-5-20250929", "gemini-2.5-pro", "grok-3"],
    sub_lm_models=["claude-haiku-4-5-20251001", "gpt-5-mini"],

    # Temperatures
    arbiter_temperature=0.1,        # low for deterministic resolution
    orchestrator_temperature=0.3,   # moderate for creative trace design
    sub_lm_temperature=0.3,         # moderate for sub-LM queries

    # Trace assignment: round_robin | random | orchestrator
    trace_assignment="round_robin",

    # Scaffolding: auto | open | guided | structured
    scaffolding_preference="auto",

    # Local inference server (vLLM-mlx, LM Studio, Ollama)
    local_server_url="http://localhost:8000/v1",
    local_model_name="default",
)

budget = RecursionBudget(
    max_depth=3,          # max recursion depth
    max_total_calls=20,   # max LLM calls across all depths
    max_cost_usd=10.0,    # cost ceiling
)
```

## Running Tests

```bash
# All unit tests (287 tests)
uv run pytest tests/ --ignore=tests/test_integration.py -v

# Integration tests (requires API keys)
uv run pytest tests/test_integration.py -v

# With coverage
uv run pytest tests/ --ignore=tests/test_integration.py --cov=rde

# Lint
uv run ruff check rde/ tests/ examples/
```

## Ablation Studies

10 ablation scripts in `examples/ablations/` for systematic evaluation:

```bash
# Same-model vs cross-model traces
uv run python -m examples.ablations.same_vs_cross_model

# Context externalization vs direct prompting
uv run python -m examples.ablations.context_externalization

# Recursive arbitration vs single-pass
uv run python -m examples.ablations.recursive_vs_single

# All ablations use shared harness: examples/ablations/harness.py
```

See `examples/dashboard.py` for a summary dashboard across ablation results.

## Benchmarks

Three benchmark suites in `rde/benchmarks/` for evaluating long-context and reasoning capabilities:

```bash
# Run OOLONG benchmark (long-context QA, 400 questions per context length)
uv run python -m rde.benchmarks.runner --benchmark oolong --context-len 1024

# Run S-NIAH benchmark (needle-in-a-haystack)
uv run python -m rde.benchmarks.runner --benchmark s_niah --context-len 1024

# Run OOLONG-Pairs benchmark (relational memory)
uv run python -m rde.benchmarks.runner --benchmark oolong_pairs --context-len 1024

# Run baselines (single model, no RDE)
uv run python -m rde.benchmarks.runner --benchmark oolong --baseline --model claude-sonnet-4-5-20250929
```

Benchmark results at 1K context with Claude Sonnet: OOLONG 80.9%, S-NIAH 100%, OOLONG-Pairs 100%.

## Training Data Pipeline

Phase 6 provides tools to distill RDE's multi-model reasoning into training data for single-model fine-tuning:

```bash
# 1. Collect training data (runs RDE on 20 diverse problems)
uv run python -m examples.training.collect_training_data

# 2. Format for provider (Gemini, Anthropic, or generic JSONL)
uv run python -m examples.training.format_for_gemini --strategy resolution

# 3. Evaluate fine-tuned model against multi-model baseline
uv run python -m examples.training.evaluate_finetuned --model tunedModels/my-model
```

Three distillation strategies:
- **Resolution**: Problem → arbiter's resolution + causal chain + shadows
- **Full Dialectic**: Problem → all perspectives + synthesis
- **Role-Specific**: (Problem + role) → individual trace output

## Project Structure

```
rde/
├── engine.py           # Main orchestration loop
├── environment.py      # Externalized context environment
├── orchestrator.py     # LLM-driven trace design
├── trace.py            # Trace execution
├── normalizer.py       # Cross-model normalization
├── arbiter.py          # Recursive causal arbitration
├── models.py           # Pydantic data models
├── cli.py              # CLI entry point
├── benchmarks/         # OOLONG, S-NIAH, OOLONG-Pairs evaluation
├── providers/          # 8 model providers + router
├── prompts/            # System prompts and templates
├── sandbox/            # Code execution (local, Modal, E2B)
├── training/           # Training data pipeline
│   ├── collector.py    # Run RDE, save results
│   ├── formatter.py    # Distillation strategies
│   ├── evaluator.py    # Fine-tuned vs baseline
│   └── models.py       # Training-specific models
└── utils/
    ├── metrics.py       # Independence metrics
    ├── extraction.py    # JSON/boxed answer extraction
    └── visualizer.py    # Call tree visualization

scripts/
└── run_vllm_mlx.sh     # Launch vLLM-mlx local server

tests/                  # 287 unit tests
examples/
├── ablations/          # 10 ablation study scripts
├── training/           # Training pipeline examples
└── dashboard.py        # Results dashboard
```

## License

See LICENSE file for details.
