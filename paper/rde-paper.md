# Recursive Dialectical Engine: Multi-Model Composition Through Causal Arbitration

## Abstract

Large language models collapse high-dimensional problems into single projections determined by their training distribution. We present the Recursive Dialectical Engine (RDE), an architecture that composes multiple independent LLM traces — each from a different model family, role, and perspective — into a single resolution through causal arbitration rather than voting. The system externalizes context as a programmable environment, enables recursive sub-arbitration on unresolved dimensions, and reports what each resolution necessarily obscures (shadows). We describe the architecture, its theoretical grounding in deliberate dimensional composition, the independence metrics used to measure trace orthogonality, and the training data pipeline for distilling multi-model reasoning into single-model fine-tuning. The implementation supports 7 model providers, 6 independence metrics, 10 ablation study configurations, and 3 distillation strategies across 185 unit tests.

## 1. Introduction

A single LLM call produces one projection of a problem — the projection determined by that model's training data, optimization objectives, and architectural choices. For well-defined problems with known solutions in the training distribution, this is sufficient. For problems requiring multiple analytical dimensions (ethical reasoning, scientific uncertainty, causal analysis), a single projection necessarily collapses dimensions that matter.

The standard response is prompt engineering: instruct the model to "consider multiple perspectives" or "think step by step." But a model prompted to consider multiple perspectives generates those perspectives from the same weight manifold. The projections are not independent — they share the same probability distribution, the same biases, the same blind spots.

RDE takes a structural approach. Instead of asking one model to simulate multiple perspectives, it runs genuinely independent traces across different model families (Claude, GPT, Gemini, Grok), each with a different role and perspective on the same externalized context. Where traces disagree, a recursive arbiter resolves through causal necessity — identifying what *must* follow from the evidence, not what the majority thinks. What the resolution necessarily obscures is reported as shadows.

### Contributions

1. An architecture for composing independent LLM projections through causal arbitration, with recursive sub-arbitration on unresolved dimensions.
2. Context externalization: the prompt as a programmable environment that traces interact with through operations (peek, search, partition), rather than receiving as raw context.
3. Six independence metrics for measuring trace orthogonality, including agreement-for-different-reasons detection.
4. A training data pipeline with three distillation strategies for fine-tuning single models on multi-model dialectical outputs.
5. An open-source implementation with 7 providers, 185 tests, and 10 ablation configurations.

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

Recursive Language Models (Qu et al., 2024) demonstrate that LLMs can use their own output as input for progressive refinement. RDE extends this with cross-model recursion: sub-problems are dispatched to independent model families, and results are composed through arbitration rather than self-refinement.

## 3. Theoretical Framework

### 3.1 Projection and Collapse

Every LLM call is a projection: a high-dimensional problem is mapped to a lower-dimensional output determined by the model's training distribution. This projection necessarily collapses some dimensions — not as an error, but as an inherent property of producing any definite output from an underspecified space.

A model trained primarily on mathematical reasoning will project ethical dilemmas through a mathematical lens. A model optimized for safety will project controversial topics through a safety-first lens. Neither projection is wrong; both are incomplete.

### 3.2 Structural Independence Through Model Diversity

When two traces share the same weights (same model, different prompts), their projections are correlated — they collapse from the same manifold. Prompt engineering can orient the projection axis but cannot escape the manifold.

Different model families (Claude, GPT, Gemini, Grok) are trained on different data distributions with different optimization objectives. Their projections are structurally independent: they collapse different dimensions differently. This is not a claim about model quality but about the geometry of their respective probability spaces.

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

The prompt is externalized as a programmable environment rather than passed as raw LLM context. Traces interact with the environment through four operations:

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

Traces execute concurrently via async gather. Each trace:

1. Receives its TraceConfig and assigned model.
2. Builds a user prompt using its context strategy.
3. Makes an independent LLM call.
4. Returns a TraceResult (raw output, extracted answer, timing, errors).

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

## 7. Discussion

### 7.1 When RDE Helps and When It Doesn't

RDE provides the most value on problems requiring multiple analytical dimensions where different models bring genuinely different strengths. Problems with well-known single-path solutions (simple math, factual lookup) gain little from multi-model composition — a single strong model is sufficient.

The architecture's value increases with:
- Problem ambiguity (multiple valid framings)
- Dimensional complexity (ethical + empirical + formal dimensions)
- Model diversity (more distinct providers = more orthogonal projections)
- Available budget (recursion can address deeper interference)

### 7.2 Fundamental Limits

RDE cannot generate genuine novelty. Every trace, every reframing, every sub-arbitration recombines patterns already in the training distributions. The system explores the framing space exhaustively but cannot escape it. Novel insights require human intervention at the shadow interface — injecting framings from outside the models' manifolds.

The system's ceiling is always determined by what the available models already encode. RLHF-collapsed models may show dramatic gains from multi-model composition (recovering dimensions suppressed during alignment). Models already trained on chain-of-thought show incremental gains. Weak models may show no gains regardless of architecture.

### 7.3 The Distillation Question

Can a single model, fine-tuned on multi-model RDE outputs, internalize cross-model diversity? Or does training on a single loss function inevitably collapse back to a single manifold?

This is the central question of Phase 6. If distillation succeeds, it suggests that multi-model reasoning patterns can be learned as a cognitive skill. If it fails, it confirms that structural independence requires structural diversity — you cannot learn to be multiple probability distributions from a single set of weights.

### 7.4 Human-in-the-Loop

The shadow report is the architectural boundary between machine and human contribution. The machine explores framing space at machine speed, composes results, and reports what it cannot address. The human evaluates shadows and either accepts the resolution or injects novel reframings.

This is not a limitation to be overcome but a design principle. The system is explicit about what it can and cannot do, and the interface for human contribution is precisely specified: shadows.

## 8. Limitations and Future Work

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
- Evaluation: Measuring "reasoning quality" on open-ended problems lacks ground truth.

## 9. Conclusion

The Recursive Dialectical Engine demonstrates that composing independent LLM projections through causal arbitration produces structurally different results than single-model reasoning or same-model ensembles. The key insight is that model diversity provides structural independence that prompt engineering alone cannot achieve — different weight distributions collapse problems differently, and composing across those collapses recovers dimensions that any single projection necessarily obscures.

The architecture makes three commitments: independence (traces cannot see each other), necessity (the arbiter demands causal chains, not votes), and honesty (shadows report what the resolution obscures). These commitments are non-negotiable because they are what make the composition meaningful. Without independence, composition degenerates into self-agreement. Without necessity, resolution degenerates into popularity. Without shadow reporting, the system cannot know what it doesn't know.

The system is itself a projection — these design choices necessarily obscure alternatives. The architecture should be held provisionally, revised as evidence accumulates, and evaluated not by whether it achieves truth but by whether it achieves the best available composition given the constraints it operates under.

## References

Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). Improving Factuality and Reasoning in Language Models through Multiagent Debate. arXiv:2305.14325.

Irving, G., Christiano, P., & Amodei, D. (2018). AI Safety via Debate. arXiv:1805.00899.

Jiang, D., Ren, X., & Lin, B. Y. (2023). LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion. ACL 2023.

Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R., ... & Shi, S. (2023). Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate. arXiv:2305.19118.

Qu, C., Dai, S., Wei, X., Cai, H., Wang, S., Yin, D., ... & Ma, J. (2024). Recursive Introspection: Teaching Language Model Agents How to Self-Improve. arXiv:2407.18219.

Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., ... & Zhou, D. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., ... & Zhou, D. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 2022.

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. NeurIPS 2023.

Zhuge, M., Liu, H., Faccio, F., Ashley, D. R., Csordas, R., Gober, A., ... & Schmidhuber, J. (2023). Mindstorms in Natural Language-Based Societies of Mind. arXiv:2305.17066.
