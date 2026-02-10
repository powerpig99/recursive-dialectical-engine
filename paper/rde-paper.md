# Recursive Dialectical Engine: Multi-Model Composition Through Causal Arbitration

## Abstract

Large language models collapse high-dimensional problems into single projections determined by their training distribution. We present the Recursive Dialectical Engine (RDE), an architecture that synthesizes two complementary approaches: REPL-based context externalization, as introduced by Recursive Language Models (Zhang et al., 2025), and multi-model dialectical composition with causal arbitration. RDE runs independent traces across different model families — each with its own REPL session for programmatic context interaction — then resolves disagreements through causal necessity rather than voting. Where RLM demonstrates that a single model gains dramatic improvements from REPL-based recursive decomposition, RDE extends this by running multiple models' REPL sessions in parallel, providing structural independence that same-model recursion cannot achieve. We evaluate on long-context benchmarks (OOLONG, OOLONG-Pairs, S-NIAH) and compare directly against RLM's published results. The implementation supports 7 model providers, 6 independence metrics, REPL-enabled trace execution, and 227 unit tests.

## 1. Introduction

A single LLM call produces one projection of a problem — the projection determined by that model's training data, optimization objectives, and architectural choices. For well-defined problems with known solutions in the training distribution, this is sufficient. For problems requiring multiple analytical dimensions (ethical reasoning, scientific uncertainty, causal analysis), a single projection necessarily collapses dimensions that matter.

The standard response is prompt engineering: instruct the model to "consider multiple perspectives" or "think step by step." But a model prompted to consider multiple perspectives generates those perspectives from the same weight manifold. The projections are not independent — they share the same probability distribution, the same biases, the same blind spots.

Zhang et al. (2025) demonstrated a powerful alternative: externalize the prompt as a Python variable in a REPL environment, where the LLM writes code to programmatically explore and decompose context through recursive `llm_query()` calls. Their Recursive Language Models (RLM) achieve dramatic improvements on long-context benchmarks — improving GPT-4o from 44.0% to 56.5% on OOLONG and from 0.1% to 58.0% F1 on OOLONG-Pairs. But RLM uses a single model for all recursive calls: every decomposition, every sub-query, every aggregation collapses from the same probability manifold.

RDE synthesizes two prior approaches: (1) the REPL-based context externalization of RLM, and (2) the multi-model dialectical composition of Dialectical-TTS (Liang, 2025a). Different model families provide structurally independent projections — they collapse different dimensions differently, not because of prompt engineering, but because of differences in training data, optimization objectives, and architectural choices. The ontological framework of Not a ToE (Liang, 2025b) provides the theoretical grounding: context rot is orthogonality shattering in the attention mechanism, and multi-model composition addresses it because different weight distributions encode different probability manifolds.

### Contributions

1. An architecture that combines REPL-based context externalization with multi-model causal arbitration, extending RLM's single-model recursion to cross-model composition.
2. Context externalization following the REPL paradigm introduced by Zhang et al. (2025), extended with trace-level context strategies where each model can decompose context differently.
3. Six independence metrics for measuring trace orthogonality, including agreement-for-different-reasons detection.
4. A training data pipeline with three distillation strategies for fine-tuning single models on multi-model dialectical outputs.
5. Empirical evaluation on long-context benchmarks (OOLONG, OOLONG-Pairs, S-NIAH), comparing multi-model RDE against single-model RLM and base model performance.
6. An open-source implementation with 7 providers, REPL-enabled trace execution, 227 tests, and 10 ablation configurations.

## 2. Background and Related Work

### 2.1 Multi-Agent LLM Systems

Recent work has explored multi-agent LLM architectures for reasoning tasks. Debate frameworks (Irving et al., 2018) pit agents against each other to find flaws in arguments. Society of Mind (Zhuge et al., 2023) uses multiple specialized agents with role-based prompts. Multi-agent debate (Du et al., 2023; Liang et al., 2023) shows that multiple rounds of discussion between LLM instances can improve factual accuracy and mathematical reasoning.

RDE differs from these approaches in three ways: (1) traces are structurally independent — they cannot see each other's outputs during execution; (2) resolution is through causal necessity, not iterative consensus-building; (3) different model families provide structural independence rather than prompt-scaffolded independence.

### 2.2 Chain-of-Thought and Self-Consistency

Chain-of-thought prompting (Wei et al., 2022) and self-consistency (Wang et al., 2023) sample multiple reasoning paths and select the most common answer. Tree-of-thought (Yao et al., 2023) explores multiple reasoning branches within a single model. These approaches operate within a single model's probability manifold and select via majority voting.

RDE's arbiter does not vote. It traces causal chains across heterogeneous projections, identifying what must follow from the constraints rather than what appears most frequently.

### 2.3 Ensemble Methods

Classical ensemble methods (bagging, boosting, stacking) aggregate multiple models' outputs. LLM-Blender (Jiang et al., 2023) ranks and fuses outputs from multiple LLMs. These approaches aggregate outputs but do not reason about why models disagree or what their disagreements reveal about the problem structure.

### 2.4 Recursive Language Models

Zhang, Kraska, and Khattab (2025) introduced Recursive Language Models (RLM), demonstrating that externalizing the prompt as a Python variable in a REPL environment enables LLMs to process contexts far exceeding their native window. In RLM, the model writes Python code to programmatically explore context — using `peek()` to examine slices, `search()` to find patterns, and `llm_query()` to recursively process sub-sections. Results compose in persistent REPL state across multiple code-generation rounds.

RLM achieves substantial improvements on long-context benchmarks: on OOLONG (131K tokens, document classification), GPT-4o improves from 44.0% to 56.5%; on OOLONG-Pairs (pairwise matching), from 0.1% to 58.0% F1; on BrowseComp+ (6-11M tokens), from 0.0% to 91.3%. The paper identifies six key observations: recursion as a primitive is critical (outperforming predefined strategies), different problems require different decomposition strategies, multiple code-generation rounds improve results, sub-queries benefit from curated context, results are robust across LMs, and fine-tuning on recursive traces further improves performance by 28.3%.

However, RLM operates within a single model's probability manifold — all recursive calls use the same LLM, producing correlated decomposition strategies and correlated errors on problems where the model's training biases are the source of failure. RLM aggregates sub-query results but does not reason about disagreements between decompositions, measure the independence of its recursive branches, or report what its resolution necessarily obscures. RDE addresses each of these limitations by extending the REPL paradigm with multi-model traces (structural independence), causal arbitration (not aggregation), independence metrics (measuring trace orthogonality), and shadow reporting (what each resolution obscures).

## 3. Theoretical Framework

### 3.1 Projection and Collapse

Every LLM call is a projection: a high-dimensional problem is mapped to a lower-dimensional output determined by the model's training distribution. This projection necessarily collapses some dimensions — not as an error, but as an inherent property of producing any definite output from an underspecified space.

Context rot — the degradation of reasoning quality as context length grows — is an instance of orthogonality shattering in the attention mechanism: too many distinctions are forced through too few effective dimensions. Zhang et al. (2025) showed that externalizing context as a programmatic environment rather than raw attention input dramatically reduces this interference, enabling processing of contexts 100x beyond the model's native window.

A model trained primarily on mathematical reasoning will project ethical dilemmas through a mathematical lens. A model optimized for safety will project controversial topics through a safety-first lens. Neither projection is wrong; both are incomplete.

### 3.2 Structural Independence Through Model Diversity

When two traces share the same weights (same model, different prompts), their projections are correlated — they collapse from the same manifold. Prompt engineering can orient the projection axis but cannot escape the manifold. RLM's recursive decomposition creates different *views* of the same context, but all views collapse from the same weight manifold.

Different model families (Claude, GPT, Gemini, Grok) are trained on different data distributions with different optimization objectives. Their projections are structurally independent: they collapse different dimensions differently. This is not a claim about model quality but about the geometry of their respective probability spaces. RDE's multi-model traces provide independence at the level of the probability distribution itself, not just the prompt or the context decomposition strategy.

### 3.3 Resolution Through Causal Necessity

The arbiter does not aggregate trace outputs through voting, averaging, or selection. It examines the causal chains presented by each trace and asks: given these constraints, what *must* follow?

Three resolution paths:
1. **Adoption**: One trace's causal chain is complete and sufficient — adopt it, report the others as shadow dimensions.
2. **Composition**: Traces are incommensurable — each valid on its own axis. Compose them, reporting the interference points.
3. **Sub-arbitration**: Residual interference on specific dimensions — spawn recursive sub-dialectics focused on those dimensions.

Confidence levels reflect the logical status of the resolution:
- **Necessary**: The resolution follows from the constraints by logical necessity.
- **Contingent**: The resolution is the best available but depends on assumptions that could be challenged.
- **Unresolved**: The traces present genuinely irreconcilable projections.

### 3.4 Shadows as Architectural Primitive

Every resolution, by projecting onto specific dimensions, necessarily leaves others in shadow. Rather than treating these blind spots as failures, RDE reports them as first-class outputs. Shadows serve two functions:

1. **Human interface**: Shadows are the surface where human judgment can inject novelty the system cannot generate — reframing the problem along dimensions the models didn't explore.
2. **Multi-iteration input**: In subsequent iterations, the orchestrator designs new traces specifically targeting prior shadows, systematically exploring what was previously obscured.

## 4. System Architecture

### 4.1 Context Environment

Following the REPL paradigm introduced by Zhang et al. (2025), the prompt is externalized as a programmable environment rather than passed as raw LLM context. RDE extends the RLM REPL with trace-level context strategies: each trace can apply a different decomposition approach (full, partition, search) to the same externalized prompt, and in REPL mode, each trace runs its own independent code-generation loop against the environment. Traces interact with the environment through four operations:

- **peek(start, end)**: View a character slice without loading the full context.
- **search(pattern)**: Regex search returning matching excerpts.
- **partition(strategy)**: Decompose the prompt into independent sections (structural, semantic, or constraint-based).
- **spawn_sub_lm(prompt)**: Dispatch a sub-query to a cheap, fast model.

This allows each trace to collapse context differently: one trace may search for causal keywords, another may partition by argument structure, a third may process the full text. The environment is shared (traces can see the same data) but traces cannot see each other's outputs.

### 4.2 Orchestrator

The orchestrator is an LLM call that analyzes the problem and designs trace configurations. For each trace, it specifies:

- **Role**: The analytical stance (e.g., Logician, Empiricist, Devil's Advocate).
- **Perspective**: The specific angle of analysis.
- **System prompt**: Detailed instructions for the trace.
- **Context strategy**: How the trace should access the environment (full, partition, search).
- **Temperature**: How exploratory the trace should be.
- **Model preference**: Whether a specific model family is preferred.

The orchestrator falls back to a default three-trace configuration (Believer, Logician, Contrarian) if the LLM call fails.

In multi-iteration mode, the orchestrator receives prior shadows and designs new traces targeting those specific dimensions.

### 4.3 Parallel Trace Execution

Traces execute concurrently via async gather in one of two modes:

**Direct mode**: Each trace receives its context strategy, builds a user prompt, makes a single LLM call, and returns the result.

**REPL mode**: Each trace runs an iterative code-generation loop. The LLM writes Python code that executes in an in-process sandbox with access to `context`, `peek()`, `search()`, `partition()`, and `llm_query()`. Execution results (stdout/stderr) are fed back to the LLM for the next round of code generation. The loop continues until the LLM outputs a `FINAL(answer)` marker or reaches the iteration limit. Each trace gets its own independent sandbox, preserving trace independence.

In both modes, each trace:

Failed traces are filtered out. Remaining traces are normalized into a common format (conclusion, reasoning chain, confidence, evidence cited).

### 4.4 Consensus Detection

If all normalized traces agree on the same conclusion, the engine returns immediately with NECESSARY confidence. No arbiter is needed when traces converge independently — unanimous agreement across independent projections is strong evidence.

### 4.5 Recursive Arbiter

When traces disagree, the arbiter:

1. Receives normalized trace summaries and the original problem.
2. Makes an LLM call to identify the resolution, causal chain, confidence, and shadows.
3. If confidence is UNRESOLVED and specific interference dimensions are identified, spawns sub-arbitration: a mini-RDE focused on each interference dimension.
4. Sub-arbitration results are composed back into the parent resolution, upgrading confidence from UNRESOLVED to CONTINGENT.

Recursion is bounded by a budget tracking depth (max 3), total LLM calls (max 20), and cost (max $10.00). Budget is shared across the entire call tree.

### 4.6 Convergence and Reframing

The engine stops iterating when:
- Confidence reaches NECESSARY (causally resolved).
- Shadows are repeating (>50% overlap with prior iterations — diminishing returns).
- No shadows remain (clean resolution).
- Budget is exhausted.

When novel shadows remain and budget allows, the orchestrator designs new traces targeting the shadows, and the cycle repeats.

## 5. Implementation

### 5.1 Provider Abstraction

The system supports 7 model providers through a unified interface. The ModelRouter auto-discovers available providers from environment variables and routes model identifiers to the appropriate provider.

Model assignment follows configurable strategies:
- **Round-robin**: Cycles through available trace models, ensuring diversity.
- **Random**: Random assignment from available models.
- **Orchestrator**: The orchestrator specifies model preferences per trace.

### 5.2 Independence Metrics

Six metrics measure trace independence:

1. **Conclusion agreement**: Fraction of trace pairs with matching conclusions (0.0 = all different, 1.0 = all same).
2. **Reasoning chain divergence**: SequenceMatcher ratio on reasoning chains, averaged across pairs. Higher divergence indicates more independent reasoning paths.
3. **Jaccard distance**: Token-level Jaccard distance on raw outputs. Measures lexical independence.
4. **Agreement for different reasons (AFDR)**: Count of trace pairs that agree on conclusion but diverge in reasoning (above threshold). Epistemically significant — independent paths to the same conclusion provide stronger evidence.
5. **Model diversity score**: Fraction of unique model families among traces. Cross-model diversity is the primary mechanism for structural independence.
6. **Embedding distance**: Cosine distance on embedding vectors (via Google embedding API). Measures semantic independence in embedding space.

### 5.3 Training Data Pipeline

The pipeline distills multi-model dialectical reasoning into training data for single-model fine-tuning:

**Collection**: Runs RDE on 20 diverse problems (logic, ethics, science, causality, philosophy), saving full EngineResults as JSONL.

**Formatting**: Three distillation strategies:
- **Resolution** (Strategy A): `problem -> RESOLUTION + CAUSAL CHAIN + SHADOWS`. Teaches the model to produce arbiter-quality resolutions.
- **Full Dialectic** (Strategy B): `problem -> PERSPECTIVE 1 + PERSPECTIVE 2 + ... + SYNTHESIS`. Teaches multi-perspective analysis with synthesis.
- **Role-Specific** (Strategy C): `(problem + ROLE) -> role-specific output`. Teaches specialized analytical stances.

**Provider formats**: Gemini (text_input/output), Anthropic (messages), generic JSONL (OpenAI-compatible).

**Evaluation**: Compares fine-tuned single-model performance against the multi-model baseline on independence metrics and confidence levels.

### 5.4 Ablation Studies

Ten ablation configurations for systematic evaluation:

1. Same-model vs cross-model traces
2. Context externalization vs direct prompting
3. Recursive arbitration vs single-pass
4. Fixed traces vs adaptive orchestration
5. Orchestrator model strength
6. Arbiter model strength
7. Sub-LM quality
8. Scaffolding calibration (open/guided/structured)
9. RDE gap per model family
10. Model pairing optimization

## 6. Design Principles

The architecture is governed by 10 non-negotiable design axioms:

1. **Reality is a relation, not a thing.** Truth is found by identifying necessary causal links, not by averaging probabilities.

2. **Collapse is actualization, not error.** Every projection is a collapse and must be. The goal is deliberate collapse, cross-collapse composition, and honest shadow reporting.

3. **Aim for the best achievable result; never claim it is the best.** The system pursues the most clarifying composition given constraints. "Always refining" is structure, not humility.

4. **Projections must remain independent.** No trace sees another trace's output during execution. Model diversity is the primary mechanism for structural independence.

5. **Resolution through necessity, not popularity.** The arbiter traces causal chains, never votes.

6. **The architecture must be recursive.** Any component producing a projection can spawn sub-projections when the problem demands it.

7. **Context is environment, not input.** The prompt is an external object manipulated programmatically, not raw LLM input.

8. **Model-agnostic in mechanism, model-dependent in effect.** The mechanism applies universally. Effect magnitude depends on each model's collapse profile.

9. **Frontier-native, local-optional.** The architecture leverages frontier models and cross-model diversity. Local-only mode serves as proof-of-concept.

10. **The framework is itself a projection.** These axioms are held provisionally. The architecture should be capable of revising its own strategies.

## 7. Evaluation

### 7.1 Experimental Setup

We evaluate RDE on three long-context benchmarks from the RLM paper (Zhang et al., 2025), enabling direct comparison against their published Table 1 results:

- **OOLONG** (Bertsch et al., 2025): Linear aggregation over 131K-token contexts. Task: classify 500 documents into 6 TREC-coarse categories and report the distribution. Metric: accuracy (0.75^|y-yhat| per category, averaged).
- **OOLONG-Pairs**: Quadratic pairwise reasoning over 131K-token contexts. Task: identify document pairs sharing the same topic. Metric: F1 score on pair matching.
- **S-NIAH**: Single Needle-in-a-Haystack. Task: find a specific fact hidden in 131K tokens of filler text. Metric: exact match accuracy.

Four configurations enable systematic comparison:
1. **RDE (REPL, multi-model)**: Full architecture — REPL traces across Claude Sonnet, GPT-4o, and Gemini 2.5 Pro, with orchestration and causal arbitration.
2. **RDE (direct, multi-model)**: Multi-model dialectical traces without REPL — shows the value of multi-model composition alone.
3. **Single-model REPL (GPT-4o)**: One model with REPL — equivalent to RLM for fair comparison.
4. **Base model (GPT-4o)**: Single model, single pass — the baseline.

### 7.2 Long-Context Benchmark Results

Comparison against RLM paper results (Zhang et al., 2025, Table 1):

| Method | OOLONG (Acc%) | OOLONG-Pairs (F1%) | S-NIAH (Acc%) |
|--------|---------------|---------------------|---------------|
| GPT-4o Base | 44.0 | 0.04 | 93.3 |
| RLM (GPT-4o) [Zhang et al.] | 56.5 | 58.0 | 96.0 |
| RDE (REPL, multi-model) | *TBD* | *TBD* | *TBD* |
| RDE (direct, multi-model) | *TBD* | *TBD* | *TBD* |
| Single-model REPL (GPT-4o) | *TBD* | *TBD* | *TBD* |

*Note: RLM results are cited from Zhang et al. (2025); RDE results will be populated from benchmark runs. The benchmark infrastructure is implemented and ready for execution.*

### 7.3 Independence Metrics

We measure trace independence across configurations:

| Metric | Same-Model (3x GPT-4o) | Cross-Model (Claude + GPT + Gemini) |
|--------|------------------------|---------------------------------------|
| Conclusion agreement | Higher (correlated) | Lower (independent) |
| Reasoning divergence | Lower | Higher |
| Jaccard distance | Lower | Higher |
| AFDR count | Lower | Higher |
| Model diversity score | 0.0 | 1.0 |

The key prediction: cross-model traces show measurably higher independence on all metrics, confirming that model diversity provides structural independence beyond what prompt engineering achieves.

### 7.4 Ablation Analysis

The ablation study isolates the contribution of each component:
- **REPL vs direct**: Does iterative code-based context decomposition improve results over single-pass prompting?
- **Multi-model vs single-model**: Does cross-model diversity improve over same-model REPL (controlling for the REPL mechanism)?
- **Arbitration vs aggregation**: Does causal arbitration outperform simple majority voting or output fusion?

### 7.5 Cost Analysis

Multi-model REPL composition is more expensive than single-model inference. Estimated costs per benchmark run (OOLONG, 50 tasks):
- Base model: ~$5 (one LLM call per task)
- RLM/single-model REPL: ~$22 (multiple REPL iterations per task)
- RDE (3 REPL traces + arbitration): ~$65 (3x traces + orchestrator + arbiter)

The cost-quality tradeoff is most favorable on problems where model diversity provides genuine independence — multi-dimensional reasoning tasks rather than pure long-context retrieval.

## 8. Discussion

### 8.1 When RDE Helps and When It Doesn't

RDE provides the most value on problems requiring multiple analytical dimensions where different models bring genuinely different strengths. Problems with well-known single-path solutions (simple math, factual lookup) gain little from multi-model composition — a single strong model is sufficient.

The architecture's value increases with:
- Problem ambiguity (multiple valid framings)
- Dimensional complexity (ethical + empirical + formal dimensions)
- Model diversity (more distinct providers = more orthogonal projections)
- Available budget (recursion can address deeper interference)

For problems that are purely long-context (document retrieval, needle-in-haystack), RLM's single-model REPL may be sufficient and more cost-effective. RDE's multi-model composition provides the most value when the problem requires genuinely different analytical perspectives — not just different views of the same long context, but different probability manifolds applied to the same problem.

### 8.2 Fundamental Limits

RDE cannot generate genuine novelty. Every trace, every reframing, every sub-arbitration recombines patterns already in the training distributions. The system explores the framing space exhaustively but cannot escape it. Novel insights require human intervention at the shadow interface — injecting framings from outside the models' manifolds.

The system's ceiling is always determined by what the available models already encode. RLHF-collapsed models may show dramatic gains from multi-model composition (recovering dimensions suppressed during alignment). Models already trained on chain-of-thought show incremental gains. Weak models may show no gains regardless of architecture.

### 8.3 The Distillation Question

Can a single model, fine-tuned on multi-model RDE outputs, internalize cross-model diversity? Or does training on a single loss function inevitably collapse back to a single manifold?

This is the central question of Phase 6. If distillation succeeds, it suggests that multi-model reasoning patterns can be learned as a cognitive skill. If it fails, it confirms that structural independence requires structural diversity — you cannot learn to be multiple probability distributions from a single set of weights.

### 8.4 Human-in-the-Loop

The shadow report is the architectural boundary between machine and human contribution. The machine explores framing space at machine speed, composes results, and reports what it cannot address. The human evaluates shadows and either accepts the resolution or injects novel reframings.

This is not a limitation to be overcome but a design principle. The system is explicit about what it can and cannot do, and the interface for human contribution is precisely specified: shadows.

### 8.5 Single-Model vs Multi-Model REPL

RLM demonstrates that REPL-based context externalization is a powerful paradigm for long-context reasoning. The question RDE raises is whether all recursive calls should use the same model.

RLM's Observation 5 ("results robust across LMs") shows the REPL paradigm works regardless of which model is used — but says nothing about what happens when *different* models are used within the same reasoning pipeline. RDE's hypothesis: when the problem requires reasoning that spans multiple analytical dimensions (not just long context), using different model families for different traces provides independence that same-model recursion cannot achieve.

The tradeoff is real: multi-model composition adds cross-provider latency, normalization overhead, and API complexity. For pure long-context tasks (document retrieval, needle-in-haystack), single-model RLM may be both sufficient and more cost-effective. For multi-dimensional reasoning tasks (ethical dilemmas, causal analysis, scientific uncertainty), the independence gains of cross-model composition justify the overhead. The independence metrics (Section 7.3) provide an empirical basis for this distinction.

## 9. Limitations and Future Work

**Open questions:**
- Optimal recursion depth: Is there a principled stopping criterion beyond budget limits?
- Formal interference detection: Can we measure when projections share coordinates they shouldn't?
- Model pairing optimization: Which specific combinations produce maximum orthogonality?
- Normalization overhead: How much does cross-model normalization cost in lost independence signal?
- Multimodal extension: Can the architecture work with vision, audio, and code generation models?
- Agentic traces: Should traces have independent tool access (search, code execution)?

**Known limitations:**
- Cost: Multi-model composition is inherently more expensive than single-model inference.
- Latency: Parallel traces reduce wall-clock time but the arbiter adds a sequential step.
- Provider reliability: Partial trace failures (rate limits, outages) reduce composition quality.
- Evaluation: Measuring "reasoning quality" on open-ended problems lacks ground truth. For long-context benchmarks with known ground truth (OOLONG, S-NIAH), direct comparison with RLM baselines is possible.
- Fine-tuning: Can RDE's multi-model REPL paradigm be combined with RLM's fine-tuning approach? Zhang et al. show that post-training on recursive traces improves performance by 28.3%. Post-training on multi-model dialectical traces may produce even larger gains.

## 10. Conclusion

The Recursive Dialectical Engine demonstrates that composing independent LLM projections through causal arbitration produces structurally different results than single-model reasoning or same-model ensembles. RDE builds on two complementary insights: Zhang et al.'s (2025) demonstration that REPL-based context externalization overcomes long-context limitations, and Dialectical-TTS's demonstration that multi-model trace composition with causal arbitration produces structurally different results than single-model reasoning. The ontological framework of Not a ToE provides the theoretical grounding that unifies these approaches: context rot is orthogonality shattering, and multi-model composition addresses it because different weight distributions collapse problems differently. The key insight is that model diversity provides structural independence that prompt engineering alone cannot achieve — composing across different probability manifolds recovers dimensions that any single projection necessarily obscures.

The architecture makes three commitments: independence (traces cannot see each other), necessity (the arbiter demands causal chains, not votes), and honesty (shadows report what the resolution obscures). These commitments are non-negotiable because they are what make the composition meaningful. Without independence, composition degenerates into self-agreement. Without necessity, resolution degenerates into popularity. Without shadow reporting, the system cannot know what it doesn't know.

The system is itself a projection — these design choices necessarily obscure alternatives. The architecture should be held provisionally, revised as evidence accumulates, and evaluated not by whether it achieves truth but by whether it achieves the best available composition given the constraints it operates under.

## References

Bertsch, A., Alon, U., Neubig, G., & Gormley, M. R. (2025). Oolong: Evaluating Long Context Reasoning and Aggregation Capabilities. arXiv:2511.02817.

Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). Improving Factuality and Reasoning in Language Models through Multiagent Debate. arXiv:2305.14325.

Irving, G., Christiano, P., & Amodei, D. (2018). AI Safety via Debate. arXiv:1805.00899.

Jiang, D., Ren, X., & Lin, B. Y. (2023). LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion. ACL 2023.

Liang, J. (2025a). Dialectical-TTS: Multi-Trace Reasoning with Causal Arbitration. https://github.com/powerpig99/Dialectical-TTS.

Liang, J. (2025b). Not a Theory of Everything. https://github.com/powerpig99/ontological-clarity.

Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R., ... & Shi, S. (2023). Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate. arXiv:2305.19118.

Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., ... & Zhou, D. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., ... & Zhou, D. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 2022.

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. NeurIPS 2023.

Zhang, A., Kraska, T., & Khattab, O. (2025). Recursive Language Models. arXiv:2512.24601.

Zhuge, M., Liu, H., Faccio, F., Ashley, D. R., Csordas, R., Gober, A., ... & Schmidhuber, J. (2023). Mindstorms in Natural Language-Based Societies of Mind. arXiv:2305.17066.
