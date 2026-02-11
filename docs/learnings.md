# Phase-by-Phase Learnings

Documenting what was built in each phase, what worked as designed, what needed adaptation, and key insights along the way.

## Phase 0: Foundation

**Planned**: Set up project structure, models, basic provider abstraction.

**Built**:
- All 17 Pydantic data models (TraceConfig, TraceResult, NormalizedTrace, ArbitrationResult, ConvergenceResult, RecursionBudget, CallTreeNode, EngineResult, ModelConfig, etc.)
- BaseProvider abstract class and LLMResponse dataclass
- Initial test infrastructure with pytest-asyncio

**Worked as designed**: Pydantic v2 models proved excellent for JSON serialization, validation, and the `.model_dump()` / `.model_dump_json()` round-trip pattern used extensively in the training pipeline (Phase 6).

**Key decision**: Using `str` enums (ConfidenceLevel, ConstraintLevel) for serialization compatibility. This paid off when the arbiter needed to parse confidence levels from JSON and when training data needed enum values as strings.

## Phase 1: Multi-Provider Infrastructure

**Planned**: Provider abstraction, ContextEnvironment, basic orchestrator, normalizer.

**Built**:
- 7 providers: Anthropic, OpenAI, Google, xAI, Kimi, MLX, LocalOpenAI (added Phase 8)
- ModelRouter with auto-discovery from environment variables
- ContextEnvironment with peek/search/partition/spawn_sub_lm
- Orchestrator with LLM-driven trace design and fallback to defaults
- TraceNormalizer with consensus detection
- CLI entry point

**What needed adaptation**:
- Provider auto-discovery: Initially considered explicit registration. Auto-discovery from API keys proved cleaner — the router checks for each key at initialization and only instantiates providers whose keys are set.
- Model identification: Needed a mapping from model name prefixes to providers (e.g., "claude-" → Anthropic, "gpt-" → OpenAI, "gemini-" → Google). This lived in the router rather than in individual providers.

**Key insight**: The `FakeProvider` test pattern (returning configurable responses per model) became the foundation for all engine-level tests. Mock at the provider level, not the HTTP level.

## Phase 2: Adaptive Orchestration & Multi-Iteration

**Planned**: Orchestrator designs problem-specific traces, multi-iteration reframing.

**Built**:
- `Orchestrator.design_traces()` — LLM call that produces TraceConfig list
- `Orchestrator.design_traces_for_iteration()` — shadow-informed reframing
- Convergence checking in Arbiter (necessary confidence, budget exhaustion, shadow repetition, no shadows)
- Multi-iteration loop in engine.run()
- Shadow history tracking across iterations

**Worked as designed**: The reframing loop works as proposed. When the orchestrator receives prior shadows, it designs new traces that target those specific dimensions. The convergence check correctly detects shadow repetition (>50% overlap) as diminishing returns.

**What needed adaptation**: The convergence check was placed in the Arbiter rather than as a separate component. This makes sense — the Arbiter has the context to evaluate whether shadows are novel or repeating, and whether confidence warrants stopping.

## Phase 3: Full Recursion

**Planned**: Traces spawn sub-traces, arbiter spawns sub-arbitration, budget tracking.

**Built**:
- `TraceExecutor.spawn_sub_dialectic()` — creates mini-RDE with child budget
- `Arbiter._sub_arbitrate()` — spawns sub-dialectic per interference dimension
- `RecursionBudget` with depth, call count, and cost tracking
- Call tree visualization (CallTreeNode hierarchy)
- Budget propagation from child to parent

**Key bug found and fixed**: Budget tracking initially didn't propagate consumption back from child to parent. A child could exhaust the budget without the parent knowing. Fixed by explicitly copying `total_calls` and `total_cost_usd` from child budget back to parent after sub-dialectic completes.

**What needed adaptation**: The proposal envisioned traces spawning sub-traces directly. Implementation routes through `TraceExecutor.spawn_sub_dialectic()` which creates a fresh `DialecticalEngine` sharing the parent's router. This avoids circular dependencies and keeps the router (with its provider connections) shared.

## Phase 4: Long Context & Code Execution

**Planned**: Semantic partitioning, code execution sandboxes.

**Built**:
- Three partition strategies: structural (paragraph-based), semantic (LLM-based), constraint (LLM-based)
- Async partition pre-computation via `env.prepare_partitions()`
- Three sandbox implementations: LocalSandbox (functional), ModalSandbox (stub), E2BSandbox (stub)
- Partition index strategy (`partition:structural:0` for specific sections)

**Key insight**: Structural partitioning (split by double-newlines) covers the majority of use cases. Semantic and constraint partitioning require an LLM call to chunk the text, which adds latency. The `prepare_partitions()` async method allows pre-computing these during the orchestrator phase rather than blocking trace execution.

**What needed adaptation**: The partition pre-computation needed to be triggered in the engine loop. Added a step (2.5) between model assignment and trace execution that scans trace configs for non-structural partition strategies and pre-computes them.

## Phase 5: Trace Independence Measurement

**Planned**: Independence metrics, ablation studies, optimization.

**Built**:
- 6 independence metrics:
  - `conclusion_agreement`: Fraction of trace pairs with matching conclusions
  - `reasoning_chain_divergence`: SequenceMatcher ratio on reasoning chains
  - `jaccard_distance`: Token-level Jaccard distance on raw outputs
  - `agreement_for_different_reasons`: Pairs that agree via different reasoning (epistemically significant)
  - `model_diversity_score`: Fraction of unique model families
  - `embedding_distance`: Cosine distance via Google embedding API
- `IndependenceReport` Pydantic model with summary generation
- CLI `--metrics` flag
- 10 ablation study scripts in `examples/ablations/`
- Results dashboard in `examples/dashboard.py`

**Key insight**: Agreement-for-different-reasons (AFDR) is the most epistemically significant metric. Two traces reaching the same conclusion through different reasoning chains provides stronger evidence than agreement alone. This metric uses a divergence threshold (default 0.3) — if reasoning chains diverge by more than 30% but conclusions match, the agreement is "for different reasons."

**What needed adaptation**: Embedding distance was made optional (requires Google API key for embedding model). All text-based metrics always compute; embeddings are opt-in via `include_embeddings=True`. The `--metrics` CLI flag enables embedding computation.

## Phase 6: Training Data Pipeline

**Planned**: Generate training data from RDE traces, distill into single model, evaluate.

**Built**:
- `CollectedResult` model wrapping EngineResult with collection metadata
- JSONL I/O for training data (save_results / load_results)
- 20 diverse training problems (logic, ethics, science, causality, philosophy)
- Three distillation strategies:
  - Resolution: `problem → RESOLUTION + CAUSAL CHAIN + SHADOWS`
  - Full Dialectic: `problem → PERSPECTIVE 1 + PERSPECTIVE 2 + ... + SYNTHESIS`
  - Role-Specific: `(problem + ROLE) → role-specific output` (one example per trace)
- Three provider formats: Gemini, Anthropic, generic JSONL
- `format_all()` pipeline with filtering (min_trace_count, require_arbitration)
- Evaluation framework: fine-tuned single-model vs multi-model baseline
- Example scripts for collection, formatting, and evaluation

**Key insight**: The full_dialectic strategy requires at least 2 traces (skips single-trace results). The resolution strategy skips failed results (confidence = unresolved with empty traces). These filters prevent degenerate training examples.

**What needed adaptation**: Originally planned as a simple formatter. Grew into a full pipeline with configurable strategies, provider formats, and filtering. The `FormattingConfig` model captures all these options cleanly.

## Phase 7: Audit, Documentation & Paper

**Audit findings**:
- No critical bugs
- 160 tests, lint clean after Phase 6
- Hardcoded temperatures in arbiter (0.1), orchestrator (0.3), sub-lm (0.3) → made configurable
- Missing dedicated test files for arbiter and trace executor → added (25 new tests)
- Missing README.md → created
- ~87% docstring coverage across the codebase

**Changes made**:
- Added `arbiter_temperature`, `orchestrator_temperature`, `sub_lm_temperature` to ModelConfig
- Wired temperatures through engine.py → Arbiter, Orchestrator, ContextEnvironment
- Created `tests/test_arbiter.py` (14 tests) and `tests/test_trace.py` (11 tests)
- Total: 185 tests, lint clean

## Phase 8: Benchmarks & Alternative Implementation Audits

**Planned**: Benchmark evaluation suite, audit Kimi and Codex alternative implementations.

**Built**:
- Benchmark suite (`rde/benchmarks/`): OOLONG (long-context QA from HuggingFace `oolongbench/oolong-synth`), S-NIAH (RULER-style needle-in-a-haystack), OOLONG-Pairs (relational memory with TREC-coarse data)
- Baseline execution path in `BenchmarkRunner` for single-model evaluation
- LaTeX answer parsing fix in OOLONG scoring
- 6 patterns ported from Kimi implementation audit:
  - JSON repair in `extraction.py` (trailing commas, single quotes, JS comments, unquoted keys)
  - Trace fallback to alternate model on provider failure
  - Per-provider cost tracking (COSTS dicts in Anthropic, OpenAI, Google providers)
  - Line-based environment operations (`peek_lines`, `search_lines`)
  - Model-family confidence calibration in normalizer (Anthropic 1.1x, OpenAI 0.9x)
- `LocalOpenAIProvider` replacing `MLXProvider` (from Codex audit):
  - httpx-based async HTTP to OpenAI-compatible local servers (vLLM-mlx, LM Studio, Ollama)
  - Continuous batching via vLLM-mlx enables ~3x speedup for parallel traces
  - `family:model` preference syntax in router (e.g., `"local:qwen3-8b"`)
  - vLLM-mlx startup script (`scripts/run_vllm_mlx.sh`)

**Key insight**: Direct MLX inference (`mlx_lm.generate()`) is synchronous and holds the GIL — it serializes all parallel traces. vLLM-mlx as an out-of-process server with continuous batching allows true concurrent trace execution. For RDE's architecture (N independent traces), this is the critical bottleneck for local inference.

**Key insight**: JSON repair covers ~80% of common LLM JSON failures (trailing commas, single quotes from Python-influenced models, JS-style comments, unquoted keys). The 4-strategy cascade in `extract_json_block()` handles code blocks → bare JSON → JSON arrays → repair.

**Benchmark baselines** (1K context, Claude Sonnet): OOLONG 80.9%, S-NIAH 100%, OOLONG-Pairs 100%.

**Changes**: 291 tests (up from 185 in Phase 7). 7 providers (Anthropic, OpenAI, Google, xAI, Kimi, LocalOpenAI, MLX deprecated). HuggingFace provider was planned but never implemented.

## Cross-Cutting Learnings

### What the proposal got right
1. **Context externalization is powerful** — Traces accessing the prompt programmatically rather than receiving it as raw context enables different collapse strategies per trace.
2. **Model diversity is the primary mechanism** — Different model families do encode genuinely different probability distributions. Cross-model composition is structurally different from same-model-different-prompt composition.
3. **Resolution through necessity** — The arbiter pattern (causal chains, not voting) produces more coherent outputs than majority-rules approaches.
4. **Shadow reporting is architecturally essential** — Knowing what the resolution obscures is as important as the resolution itself.

### What needed rethinking
1. **Temperature defaults matter** — The arbiter needs low temperature (0.1) for deterministic resolution; the orchestrator needs moderate temperature (0.3) for creative trace design. These were initially hardcoded and needed to be configurable.
2. **Fallback paths are critical** — Every LLM-dependent component needs a non-LLM fallback. The arbiter falls back to longest-chain selection. The orchestrator falls back to default traces. Without these, any single API failure would crash the pipeline.
3. **JSON extraction from LLMs is fragile** — LLMs wrap JSON in markdown code blocks, sometimes with extra text before/after. The `extract_json_block()` utility handles code blocks, bare JSON, and JSON arrays, with each pattern discovered through testing.

### Testing patterns that worked
1. **FakeProvider at the provider level** — Mock the provider interface, not individual API calls. This allows testing the full pipeline without any API keys.
2. **Monkeypatch API keys** — Tests delete all API keys from environment to prevent accidental real API calls.
3. **Synthetic data builders** — Helper functions like `_make_collected_result()` and `_make_traces()` that produce valid test data with configurable parameters.

### Architecture decisions validated
1. **Pydantic v2 throughout** — JSON serialization, validation, `.model_dump()` consistency across all data boundaries.
2. **Async from the start** — Every provider, the engine, the arbiter, and the training collector are async. This enables parallel trace execution naturally.
3. **Provider auto-discovery** — Setting API keys and having the router figure out what's available is the right UX. No configuration files needed.
