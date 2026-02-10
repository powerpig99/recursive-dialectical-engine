# RDE Architecture

## Overview

The Recursive Dialectical Engine composes multiple independent LLM projections into a single resolution through causal arbitration. The architecture implements a pipeline where each stage preserves trace independence and reports what the resolution necessarily obscures (shadows).

## Component Diagram

```
                    ┌──────────────┐
                    │  User Prompt │
                    └──────┬───────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  ContextEnvironment    │
              │  (externalized prompt) │
              │                        │
              │  peek() / search()     │
              │  partition() / sub_lm()│
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │     Orchestrator       │
              │  (LLM-driven design)   │
              │                        │
              │  → TraceConfig[]       │
              │  (roles, perspectives, │
              │   context strategies)  │
              └────────────┬───────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Trace A  │ │ Trace B  │ │ Trace C  │
        │ Claude   │ │ Gemini   │ │ Grok     │
        │ Believer │ │ Logician │ │Contrarian│
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │            │            │
             ▼            ▼            ▼
              ┌────────────────────────┐
              │    TraceNormalizer     │
              │                        │
              │  → NormalizedTrace[]   │
              │  (conclusion, chain,   │
              │   confidence, evidence)│
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Consensus Check      │
              │                        │
              │  All agree? → DONE     │
              │  (necessary confidence)│
              └────────────┬───────────┘
                           │ (disagreement)
                           ▼
              ┌────────────────────────┐
              │    Recursive Arbiter   │
              │                        │
              │  → ArbitrationResult   │
              │  (resolution, causal   │
              │   chain, shadows,      │
              │   confidence)          │
              │                        │
              │  Can spawn sub-arb     │
              │  on interference dims  │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Convergence Check     │
              │                        │
              │  Stop if:              │
              │  - necessary conf      │
              │  - budget exhausted    │
              │  - shadows repeating   │
              │  - no shadows          │
              │                        │
              │  Continue if:          │
              │  - novel shadows exist │
              └────────────┬───────────┘
                           │ (novel shadows)
                           ▼
                   Reframe & Iterate
                   (orchestrator targets
                    unresolved shadows)
```

## Data Flow

### Models and Their Relationships

```
ModelConfig
  │
  ├── orchestrator_model ──→ Orchestrator
  ├── arbiter_model ───────→ Arbiter
  ├── trace_models[] ──────→ ModelRouter.assign_trace_models()
  ├── sub_lm_models[] ────→ ContextEnvironment.spawn_sub_lm()
  └── temperatures ────────→ each component

TraceConfig (designed by Orchestrator)
  │  role, perspective, system_prompt
  │  context_strategy, temperature
  │  model_preference
  │
  ▼
TraceResult (produced by TraceExecutor)
  │  trace_id, role, model_used
  │  raw_output, extracted_answer
  │  error, latency_ms, token_usage
  │
  ▼
NormalizedTrace (produced by TraceNormalizer)
  │  conclusion, reasoning_chain
  │  confidence, model_family
  │
  ▼
ArbitrationResult (produced by Arbiter)
  │  resolution, causal_chain
  │  confidence (necessary|contingent|unresolved)
  │  shadows[], interference_detected[]
  │  traces_adopted[], traces_rejected[]
  │
  ▼
EngineResult (final output)
     resolution, confidence, shadows
     causal_chain, iterations
     trace_results[], normalized_traces[]
     arbitration, consensus_reached
     call_tree, total_latency_ms
```

## Provider Abstraction

```
BaseProvider (abstract)
  │
  ├── AnthropicProvider  → Claude models
  ├── OpenAIProvider     → GPT models
  ├── GoogleProvider     → Gemini models
  ├── XAIProvider        → Grok models
  ├── KimiProvider       → Moonshot models
  ├── HuggingFaceProvider → HF Inference API
  └── MLXProvider        → Local quantized models

ModelRouter
  ├── Auto-discovers providers from API keys
  ├── Routes model identifiers to providers
  ├── Assigns trace models (round_robin / random)
  └── Manages provider lifecycle
```

## Recursion Model

The engine supports recursive reasoning at two levels:

### 1. Multi-Iteration Reframing (engine.run)

When `max_iterations > 1`, the engine loops:
1. Design traces targeting current shadows
2. Execute traces
3. Arbitrate disagreements
4. Check convergence
5. If novel shadows remain and budget allows, loop with shadows as reframing input

### 2. Sub-Dialectic Spawning (trace.spawn_sub_dialectic)

When the arbiter detects unresolved interference on specific dimensions:
1. For each interference dimension, spawn a mini-RDE
2. Mini-RDE runs with child budget (depth + 1)
3. Sub-resolutions composed back into parent arbitration
4. Budget consumption propagated to parent

### Budget Tracking (RecursionBudget)

```
RecursionBudget
  max_depth: 3         ← limits recursion depth
  max_total_calls: 20  ← limits total LLM calls across all depths
  max_cost_usd: 10.0   ← cost ceiling

  can_recurse() → bool
  child_budget() → RecursionBudget (depth + 1, shared counters)
  record_call(cost_usd) → updates totals
```

## Context Strategies

The ContextEnvironment supports multiple ways for traces to access the prompt:

| Strategy | Format | Behavior |
|----------|--------|----------|
| `full` | `"full"` | Pass entire prompt as-is |
| `partition` | `"partition:structural"` | Split by paragraphs, join all parts |
| `partition+index` | `"partition:structural:0"` | Return only the Nth partition |
| `search` | `"search:pattern"` | Regex search, return matching excerpts |

Partition strategies:
- **structural**: Split by double-newlines (paragraphs)
- **semantic**: LLM-based semantic chunking (async, pre-computed)
- **constraint**: LLM-based constraint decomposition (async, pre-computed)

## Key Design Decisions

1. **Context externalization**: The prompt is never passed directly as LLM context. Traces interact with it programmatically, allowing each trace to collapse context differently.

2. **Resolution through necessity, not voting**: The arbiter traces causal chains to determine what must follow, rather than counting votes or averaging.

3. **Model diversity as structural independence**: Different model families encode different probability manifolds. Cross-model traces achieve structural orthogonality, not just prompt-scaffolded orthogonality.

4. **Shadow reporting**: Every resolution reports what it necessarily obscures. Shadows are the interface for human intervention and multi-iteration reframing.

5. **Configurable temperatures**: Arbiter (0.1 default, deterministic), Orchestrator (0.3, moderate creativity), Sub-LM (0.3, moderate). All configurable via ModelConfig.
