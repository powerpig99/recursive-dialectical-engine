# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Recursive Dialectical Engine (RDE) is an AI reasoning architecture that unifies two approaches:
- **Dialectical-TTS** (multi-trace reasoning with causal arbitration)
- **Recursive Language Models** (recursive context management via externalized REPL)

It solves two LLM structural failures: **context rot** (orthogonality shattering in attention) and **probabilistic collapse** (single-pass can't sustain independent projections for logical reasoning).

**Status**: Archived. Superseded by [Context Engine](https://github.com/powerpig99/context-engine). Core engine, 7 providers, benchmarks, training pipeline, and ablation studies are all functional. 291 unit tests, lint clean. Local benchmark validation complete — REPL implementation gap identified (see Phase 8 notes).

**Prior work**: Evolves [Dialectical-TTS](https://github.com/powerpig99/Dialectical-TTS). Informed by [RLM (Zhang et al.)](https://arxiv.org/abs/2512.24601) and [Not a ToE](https://github.com/powerpig99/ontological-clarity).

## Architecture

Core execution flow:
```
User Prompt → ContextEnvironment (REPL) → Orchestrator → N Independent Traces (parallel, cross-model)
    → TraceNormalizer → RecursiveArbiter (causal necessity, not voting) → Resolution + Shadow Report
```

### Key Components

| Component | File | Role |
|-----------|------|------|
| `ContextEnvironment` | `rde/environment.py` | Python REPL that externalizes the prompt. Traces interact via `peek`, `peek_lines`, `search`, `search_lines`, `partition`, `spawn_sub_lm` — never via direct prompt injection |
| `RootOrchestrator` | `rde/orchestrator.py` | Analyzes problem structure, designs N trace configurations (not fixed at 3). Chooses constraint level: open/guided/structured |
| `Trace` | `rde/trace.py` | Independent LM call with own system prompt, model, context strategy. Can recursively spawn sub-traces (sub-dialectics) |
| `RecursiveArbiter` | `rde/arbiter.py` | Resolves traces through Logic of Necessity. Detects interference. Spawns sub-arbitration on unresolved dimensions. Reports shadows |
| `TraceNormalizer` | `rde/normalizer.py` | Makes heterogeneous model outputs comparable (not agreeable) before arbitration |
| Providers | `rde/providers/` | Abstraction over Anthropic, OpenAI, Google, xAI, Kimi, LocalOpenAI (vLLM-mlx) APIs. Per-provider cost tracking. MLX (deprecated) |
| Sandbox | `rde/sandbox/` | Isolated REPL execution (Modal, E2B, or local subprocess) |

### Multi-Model Architecture (Core Design Principle)

Model diversity IS projection independence. Different model families encode different probability manifolds — this is the mechanism for genuine trace orthogonality.

- **Orchestrator/Arbiter**: Strongest available (Opus, GPT-5) — hardest cognitive tasks
- **Traces**: Diverse frontier models (Claude, GPT, Gemini, Qwen, Kimi) — independence matters more than raw capability
- **Sub-LM calls**: Cheapest adequate (Haiku, GPT-5-mini) — high volume, low complexity

## Tech Stack

- **Runtime**: Python 3.11+
- **LM Access**: Anthropic, OpenAI, Google, xAI, Kimi native SDKs + httpx for local
- **Local inference**: vLLM-mlx (OpenAI-compatible server on Apple Silicon) via `LocalOpenAIProvider`. Also works with LM Studio, Ollama, or any OpenAI-compatible endpoint. Legacy `MLXProvider` (deprecated) for direct in-process MLX
- **REPL Sandbox**: Local subprocess (functional), Modal / E2B (stubs)
- **Orchestration**: `asyncio` + provider async clients
- **Benchmarks**: OOLONG (long-context QA), S-NIAH (needle-in-a-haystack), OOLONG-Pairs (relational memory)
- **Testing**: pytest + pytest-asyncio, 291 unit tests

## Design Axioms (Non-Negotiable)

These govern all implementation decisions:

1. **Resolution through necessity, not popularity** — the Arbiter never votes; it traces causal chains
2. **Projections must remain independent** — traces cannot see each other's outputs during execution
3. **Context is environment, not input** — the prompt is an external object manipulated programmatically
4. **Collapse is actualization, not error** — every projection is a collapse; compose across collapses deliberately and report shadows honestly
5. **The architecture must be recursive** — any component producing a projection can spawn sub-projections
6. **Frontier-native, local-optional** — full architecture uses frontier models; local mode is for dev/test backward compatibility
7. **The framework is itself a projection** — hold all design decisions provisionally

## Ontological Grounding

The architecture derives from the Not a ToE framework. Key mapping:
- **Context rot** = orthogonality shattering in attention (2^N distinctions crammed into N coordinates)
- **Traces** = independent projections onto reasoning axes
- **Arbiter** = resolution through causal necessity (Logic of Necessity)
- **Shadows** = what each resolution necessarily obscures (every revelation deceives)
- **Iteration** = "stepping outside" — recursive arbitration is the repeated movement, never arrival

## Implementation Phases

1. **Phase 0**: Foundation — Pydantic data models, BaseProvider, test infrastructure
2. **Phase 1**: Multi-provider infrastructure (7 providers), ContextEnvironment, Orchestrator, Normalizer, CLI
3. **Phase 2**: Adaptive orchestration — LLM-driven trace design, multi-iteration reframing, convergence checking
4. **Phase 3**: Full recursion — sub-dialectic spawning, sub-arbitration, budget tracking, call tree visualization
5. **Phase 4**: Long context & code execution — partition strategies, sandbox implementations
6. **Phase 5**: Trace independence measurement — 6 metrics, ablation studies, results dashboard
7. **Phase 6**: Training data pipeline — collection, distillation (3 strategies), evaluation
8. **Phase 7**: Audit & documentation — configurable temperatures, dedicated test files, README
9. **Phase 8**: Benchmarks & alternative implementation audits
   - OOLONG, S-NIAH, OOLONG-Pairs benchmark suite (`rde/benchmarks/`)
   - Kimi implementation audit: ported JSON repair, trace fallback, cost tracking, line-based env ops, confidence calibration
   - Codex implementation audit: replaced MLXProvider with LocalOpenAIProvider (vLLM-mlx), added `family:model` preference syntax, vLLM-mlx startup script
   - Local benchmark infrastructure: 5 configs (`local_vanilla`, `local_repl`, `local_dialectical_same`, `local_dialectical_repl_same`, `local_dialectical_2iter`), overnight runner, analysis scripts
   - **Local benchmark results** (Qwen3-8B-MLX-4bit, 50 tasks):
     - Vanilla: OOLONG 66%, S-NIAH 100%, OOLONG-Pairs 10%
     - REPL: OOLONG 16%, S-NIAH 20%, OOLONG-Pairs 14% — catastrophic regression
     - Dialectical 3x: OOLONG 45%, S-NIAH 100% — worse than vanilla at scale
   - **REPL gap analysis**: 10 architectural differences from RLM repo identified (code block extraction, system prompt quality, per-iteration guidance, FINAL_VAR, max iterations, etc.)
   - **Next**: Reproduce RLM results with their actual framework on Qwen3-8B before fixing our REPL

## Development Workflow

Multi-agent development mirrors RDE's own architecture:
- **Claude Code**: Architecture, orchestrator, arbiter, system prompts
- **OpenAI Codex**: ContextEnvironment, sandbox, async infrastructure
- **Kimi Coder**: Trace execution, routing, provider abstraction

Agents work against this spec independently — no real-time coordination needed.
