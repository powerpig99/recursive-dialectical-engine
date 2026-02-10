# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Recursive Dialectical Engine (RDE) is an AI reasoning architecture that unifies two approaches:
- **Dialectical-TTS** (multi-trace reasoning with causal arbitration)
- **Recursive Language Models** (recursive context management via externalized REPL)

It solves two LLM structural failures: **context rot** (orthogonality shattering in attention) and **probabilistic collapse** (single-pass can't sustain independent projections for logical reasoning).

**Status**: Pre-implementation. The specification lives in `recursive-dialectical-engine-proposal.md`. All architectural decisions, system prompts, evaluation plans, and phased implementation details are in that document.

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
| `ContextEnvironment` | `rde/environment.py` | Python REPL that externalizes the prompt. Traces interact via `peek`, `search`, `partition`, `spawn_sub_lm` — never via direct prompt injection |
| `RootOrchestrator` | `rde/orchestrator.py` | Analyzes problem structure, designs N trace configurations (not fixed at 3). Chooses constraint level: open/guided/structured |
| `Trace` | `rde/trace.py` | Independent LM call with own system prompt, model, context strategy. Can recursively spawn sub-traces (sub-dialectics) |
| `RecursiveArbiter` | `rde/arbiter.py` | Resolves traces through Logic of Necessity. Detects interference. Spawns sub-arbitration on unresolved dimensions. Reports shadows |
| `TraceNormalizer` | `rde/normalizer.py` | Makes heterogeneous model outputs comparable (not agreeable) before arbitration |
| Providers | `rde/providers/` | Abstraction over Anthropic, OpenAI, Google, OpenRouter APIs |
| Sandbox | `rde/sandbox/` | Isolated REPL execution (Modal, E2B, or local subprocess) |

### Multi-Model Architecture (Core Design Principle)

Model diversity IS projection independence. Different model families encode different probability manifolds — this is the mechanism for genuine trace orthogonality.

- **Orchestrator/Arbiter**: Strongest available (Opus, GPT-5) — hardest cognitive tasks
- **Traces**: Diverse frontier models (Claude, GPT, Gemini, Qwen, Kimi) — independence matters more than raw capability
- **Sub-LM calls**: Cheapest adequate (Haiku, GPT-5-mini) — high volume, low complexity

## Tech Stack

- **Runtime**: Python 3.11+
- **LM Access**: OpenAI, Anthropic, Google, OpenRouter clients
- **REPL Sandbox**: Modal / E2B / Prime Intellect
- **Orchestration**: `asyncio` + provider async clients
- **Routing**: OpenRouter / LiteLLM
- **Local fallback**: MLX (Apple Silicon) — dev/test only

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

1. **Phase 1** (Weeks 1-2): Multi-provider infrastructure + ContextEnvironment
2. **Phase 2** (Week 3): Adaptive orchestration (problem-specific traces)
3. **Phase 3** (Week 4): Full recursion (sub-traces, sub-arbitration across model boundaries)
4. **Phase 4** (Week 5): Long-context integration (RLM-style)
5. **Phase 5** (Week 6): Trace independence measurement & optimization
6. **Phase 6** (Week 7+): Post-training exploration (distilling multi-model reasoning)

## Development Workflow

Multi-agent development mirrors RDE's own architecture:
- **Claude Code**: Architecture, orchestrator, arbiter, system prompts
- **OpenAI Codex**: ContextEnvironment, sandbox, async infrastructure
- **Kimi Coder**: Trace execution, routing, provider abstraction

Agents work against this spec independently — no real-time coordination needed.
