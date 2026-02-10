# Recursive Dialectical Engine (RDE)

## Research Proposal & Implementation Specification

**Author:** Jing Liang (@powerpig)
**Date:** February 2026
**Evolves:** [Dialectical-TTS](https://github.com/powerpig99/Dialectical-TTS)
**Informed by:** [Recursive Language Models (Zhang et al., arXiv:2512.24601)](https://arxiv.org/abs/2512.24601), [Not a ToE](https://github.com/powerpig99/ontological-clarity)

---

## 1. Motivation

### 1.1 The Core Problem

Large Language Models operate as single-pass projection machines. A prompt enters, attention distributes across the full context, and a response emerges. This architecture suffers from two structural failures:

1. **Context rot**: As context length grows, the model's effective reasoning degrades—not because information is lost, but because too many distinctions are forced through too few effective dimensions. The interference pattern (spurious correlations between unrelated context elements) overwhelms the signal.

2. **Probabilistic collapse**: When faced with problems requiring causal reasoning, LLMs default to frequency-based pattern matching (System 1). They "remember" answers rather than deriving them. The single forward pass cannot sustain the independent projections needed for genuine logical arbitration.

### 1.2 Prior Work: Dialectical-TTS

The original [Dialectical-TTS](https://github.com/powerpig99/Dialectical-TTS) addressed the second problem by orchestrating three independent cognitive traces:

- **The Believer (Intuition)**: Standard generation representing training bias
- **The Logician (Validation)**: Forced atomic decomposition with constraint checking
- **The Contrarian (Red Team)**: Explicit adversarial assumption that the intuitive answer is a trap

An **Arbiter** then synthesizes—not by majority vote, but by checking which argument adheres to the **Logic of Necessity** given the constraints. This is structurally sound: it maintains independent projections and resolves through causal necessity rather than averaging.

### 1.3 The RLM Breakthrough

Recursive Language Models (Zhang, Kraska, Khattab — MIT CSAIL, Dec 2025) solve the first problem through a complementary mechanism:

- The prompt is **externalized** as a variable in a Python REPL environment
- The root LM **programmatically explores** the context (peek, partition, grep, regex)
- The LM **recursively spawns sub-LM calls** on context partitions
- Results compose in the persistent REPL environment

Key results:
- Processes inputs up to **100x beyond model context windows**
- **Outperforms** vanilla frontier LLMs and common long-context scaffolds on diverse tasks
- Post-trained RLM-Qwen3-8B outperforms base Qwen3-8B by **28.3%** on average
- Prime Intellect identifies context folding via RLMs as a major research direction

### 1.4 The Structural Convergence

Both approaches address the same underlying mechanism: **dimensional collapse creates interference that destroys information**.

| Dimension | Dialectical-TTS | RLM | Convergence Point |
|-----------|----------------|-----|-------------------|
| What collapses | Reasoning traces | Context window | Distinctions forced through insufficient dimensions |
| How it's addressed | Multiple independent traces | Recursive context partitioning | Preserve orthogonality of projections |
| Resolution method | Causal arbitration (not voting) | Compositional aggregation in REPL | Let each projection operate on its own terms |
| Limitation | Fixed 3-trace architecture | No dialectical reasoning structure | Each solves half the problem |

The Recursive Dialectical Engine unifies both: **recursive context management** meets **dialectical reasoning arbitration**.

---

## 2. Theoretical Foundation

### 2.1 Ontological Grounding

From the **Not a ToE** ([repo](https://github.com/powerpig99/ontological-clarity)):

> *Everything is layered projections of the infinite-dimensional orthogonal binary hyperspace from Nothing—the infinitely self-referencing Contradiction.*

Everything else in this section is derived from that single line. "Ontological Clarity" is the name of the practical skill built from the Not a ToE. The relevant derivations:

> **Binary Hyperspace**: The Contradiction's infinite orthogonal modes of self-distinction. Each basis vector marks one pure distinction, negates all others. No overlap, no interference in the ground.

> **Projection**: Everything experienced is projection from this hyperspace onto finite dimensions. Distinctions survive (cardinality preserved) but geometry distorts.

> **Orthogonality Shattering**: 2^N distinctions crammed into N coordinates → spurious correlations, interference, entanglement.

An LLM's context window is a finite-dimensional projection space. As tokens accumulate, the effective dimensionality saturates. Distinctions that were orthogonal in the "ground truth" of the problem begin to interfere. This isn't a bug in attention—it's a structural consequence of projection.

**Context rot is orthogonality shattering in the attention mechanism.**

Larger context windows help—more N gives more room, but interference dominates nonetheless; more dimensions means more room for a greater diversity of interferences, not their absence. Better attention architectures help. Recursive context management helps. Multi-trace composition helps. None of them eliminates interference—not alone, not in combination—because 2^N distinctions outpacing N dimensions is not a problem to be solved but a condition of projection itself. The goal is not non-interference but *better* interference: untangling existing interferences as much as possible so the collapses that remain are more clarifying. Each approach contributes to this untangling along its own axis, with no guarantee and no arrival. The RDE contributes along its axis by:
1. **Not forcing all distinctions through one projection** (RLM's recursive decomposition)
2. **Maintaining independent reasoning traces that each project onto their own axes** (Dialectical-TTS's multi-trace architecture)
3. **Resolving through causal necessity rather than dimensional averaging** (the Arbiter's Logic of Necessity)

### 2.2 Design Principle: Projection Without Collapse

The Analytical Method from the framework:

1. **Trace projections as-is**: Each trace/partition operates on its own terms with its own metric
2. **Identify where collapse creates confusion**: Where is one projection being forced to do another's work?
3. **Dissolve the collapse**: Let each be what it is; resolve through necessity, not averaging

This translates directly to architecture:

- Each recursive sub-call is a **projection** — it should operate on its own partition without interference from others
- The dialectical traces are **independent projections onto reasoning axes** — they should not share context that creates spurious agreement
- The Arbiter applies **causal necessity** — which projection survives because it must, not because it's popular

### 2.3 Collapse as Actualization

The standard narrative frames test-time compute as *adding* reasoning capability to models. A subtler version frames it as *de-inhibition*—removing constraints so latent capability can surface. Both framings contain a hidden assumption: that collapse is the problem and less collapse is better.

But the framework says otherwise: "To reveal is to make distinct. To make distinct is to obscure what the distinction excludes. Illumination casts shadow by the same act."

**Collapse is not the enemy. Collapse is how potentiality becomes actuality.** Without projection from hyperspace onto finite dimensions, nothing manifests. A model’s forward pass is a collapse. Each trace is a collapse. The Arbiter’s resolution is a collapse. Every system prompt, every architectural choice, every decomposition strategy—all collapses. They must be. That’s how anything gets produced at all.

The distortion doesn’t come from collapsing. It comes from two specific errors:

1. **Mistaking the collapse for the territory**: Claiming the projection IS the thing rather than A projection of it. A single forward pass produces a result and the system treats it as the answer, rather than as one projection that necessarily obscures what it excludes.

2. **Unexamined collapse**: Collapsing without awareness of what the collapse privileges and what it suppresses. Single-pass inference doesn’t choose its collapse deliberately—it’s whatever the attention mechanism happens to produce under the combined pressure of all tokens at once.

**The RDE’s role is therefore not to minimize collapse, but to collapse deliberately, compose across multiple deliberate collapses, and never claim the composition is more than the best achievable projection given the constraints.**

This reframes every architectural decision:

| Decision | "Minimize Collapse" Frame (incomplete) | "Deliberate Collapse" Frame (correct) |
|----------|---------------------------------------|--------------------------------------|
| Multiple traces | "Remove the single-pass bottleneck" | "Generate multiple deliberate collapses that illuminate different facets; compose them for the most clarifying result" |
| Context externalization | "Remove attention-collapse" | "Allow each trace to collapse the context in its own way, deliberately, rather than forcing one undifferentiated collapse" |
| Cross-model diversity | "Different inhibition profiles release different capabilities" | "Different model families collapse differently—their collapses illuminate different facets of the problem" |
| Trace system prompts | "Open channels, don’t constrain" | "Choose the collapse that best illuminates this axis—sometimes structured, sometimes open, always deliberate" |
| Recursive arbitration | "Resolution channel for causal necessity" | "A further collapse that composes prior collapses; it produces clarity AND new shadows, which recursive levels can address" |

**A note on vocabulary**: Throughout this proposal, *collapse*, *projection*, *interference*, *scaffolding*, and *framing* are used in different contexts, but they name the same underlying process—the act of distinction that actualizes by including and excluding simultaneously. "Collapse" emphasizes that dimensionality reduces. "Projection" emphasizes the geometric relationship. "Interference" emphasizes what happens when multiple collapses share coordinates. "Scaffolding" emphasizes deliberate human-designed collapse. "Framing" emphasizes the interpretive choice. These are not different mechanisms. They are different facets of the single process by which anything manifests at all.

### 2.4 Deliberate Collapse and Honest Limits

The RDE operates on three structural mechanisms:

1. **Context externalization**: Rather than forcing one undifferentiated collapse through a single attention pass, allow each trace to interact with context programmatically and collapse it in its own way.

2. **Trace independence**: Multiple channels that cannot see each other during execution. Each produces its own deliberate collapse. The value is in the *composition* of different collapses, not in any individual one.

3. **Resolution through necessity**: The Arbiter composes trace outputs by tracing what must follow from the constraints—not by averaging collapses (which produces mush) or selecting one collapse (which discards the others’ illumination). The resolution itself is a collapse, and the Arbiter should report what it necessarily obscures.

**The Orchestrator’s job is not "minimize scaffolding" or "maximize freedom." It is: given this problem and these models, which specific collapses will compose into the most clarifying result?**

Sometimes that means highly structured traces (Believer/Logician/Contrarian) because the problem has known axes where deliberate adversarial collapse is productive. Sometimes it means open traces because the model’s own collapse structure is better than anything the designer would prescribe. Sometimes it means a hybrid. The criterion is not "how little did we constrain?" but **"how much clarity did the composition of these specific collapses produce, and how honestly did we report what the composition obscures?"**

This connects to the framework’s epistemological core: "Pursue understanding relentlessly; hold it provisionally; offer it as *a* projection, not *the* truth." The RDE pursues the best possible result given the constraints. It does not claim the result is the best possible result in absolute terms. The Arbiter’s output should always include what was necessarily left in shadow.

**Implication for implementation**: The trace system prompt library should range from highly structured to maximally open. The Orchestrator selects based on what serves clarity for this specific problem, not on a prior commitment to either end of the spectrum. The Arbiter’s output format must include a `shadows` field—what the resolution necessarily obscures or leaves unexamined.

### 2.5 Fundamental Limit: Novelty Requires Indeterminism

The RDE recursively explores the space of possible framings (collapses) and composes the most clarifying result from what it finds. This is analogous to **recursive learning with fast weights**—the system updates its decomposition strategy within a single session based on what each iteration reveals, without persistent training. Each iteration changes the framing surface itself, not just the position on a fixed surface.

But this exploration has a structural ceiling: **the system cannot inject novelty**. Every reframing it generates is a recombination of patterns already encoded in the models' weight manifolds. The models were trained on human-generated data and optimized toward human-evaluated objectives. Their projection space is vast but bounded by what was in the training distribution.

This is not a limitation to be engineered around. It is a fundamental property of deterministic computation operating on fixed parameters. Novelty—genuine reframing that comes from outside the existing projection space—requires an indeterministic source. In practice, that source is human ingenuity: the capacity to see what no model in the trace pool encodes, to ask a question none of the models would generate, to reframe in a direction that doesn't exist in any training distribution.

**The RDE is therefore a human-in-the-loop exploration accelerator:**

```
┌────────────────────────────────────────────────────────┐
│  MACHINE: Rapidly explore framing/collapse space     │
│  → Compose best achievable result                     │
│  → Report what remains in shadow                       │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  HUMAN: Evaluate result + shadows                     │
│  → Accept (sufficient for purpose)                     │
│  → Inject novel reframing (what no model could see)    │
│  → Adjust parameters and re-explore                    │
└───────────────────────────┬────────────────────────────┘
                            │ (if novel reframing)
                            ▼
                    Back to machine exploration
```

The machine’s value is **acceleration**: it can explore thousands of framings faster than a human could explore ten, within its finite projection space. The human’s value is **novelty**: they operate in the first infinity—potentiality within form, inexhaustible and extensible—and can inject framings that exist nowhere in any model’s weight manifold. Neither is sufficient alone, and neither reaches the second infinity (the Contradiction’s self-referential depth, which is ontologically prior to both). The RDE’s architecture should make the handoff surface—the shadow report—as rich and actionable as possible, because that’s where the human’s contribution from a deeper level of potentiality enters.

This clarifies the relationship between the RDE, the models, and the human: the RDE doesn’t make models smarter. It makes the finite space of framings available from those models explorable at machine speed. The ceiling of what the RDE can find is always determined by what the models already encode. Raising that ceiling requires either better models (more in the finite weights) or human intervention (novelty from the first infinity—potentiality that the finite machine cannot generate). Even the human’s contribution, however vast, remains a projection within form. The absolute ground from which all of this projects is not reachable by any of them.


### 2.6 Model Agnosticism and Model Dependence

The RDE is **model-agnostic in mechanism** but **model-dependent in effect**.

The mechanism (externalize context, independent traces, resolve through necessity) applies to any sufficiently capable language model. But the *magnitude and character* of improvement depends on each model’s specific relationship to collapse:

- A model with heavy RLHF collapses toward hedged, committee-style outputs. The RDE’s independent traces allow it to collapse along its *actual reasoning axes* rather than its safety-trained output distribution. The improvement may be dramatic.
- A model already trained with extensive chain-of-thought has more deliberate internal collapse. The RDE’s additional collapses compose with what the model already does, producing incremental rather than dramatic gains.
- A model with a small or poorly-trained weight space has less to project in any direction. No amount of deliberate collapse composition can surface capability that isn’t there.
- Different model families collapse differently because they were trained on different data, with different objectives, by different architects making different value choices. These differences are not noise—they are the source of genuine projection diversity.

This means the RDE functions as a **diagnostic instrument** as well as a reasoning tool. The gap between a model’s single-pass performance and its RDE-augmented performance (the RDE-gap) measures not "how much capability was inhibited" but **how much additional clarity becomes available when the model’s collapse is composed with other deliberate collapses rather than standing alone**.

This measurement has implications beyond the RDE itself:
- For model developers: the RDE-gap reveals how much of your model’s trained capability actualizes through single-pass inference vs. through composed projection
- For model comparison: two models with similar single-pass benchmarks may have very different RDE-gaps, revealing different collapse profiles—one may be producing a single high-quality collapse, the other may be averaging across many internal directions
- For alignment research: the RDE-gap on safety-relevant vs. safety-neutral tasks reveals whether alignment training is collapsing specifically toward safety or collapsing broadly in ways that sacrifice general capability as collateral

### 2.7 The Unifying Insight: Everything Is Context Engineering

Every section above—collapse as actualization, deliberate framing, model diversity, fundamental limits, human-in-the-loop—resolves into a single insight:

**Predicting the next token is "all you need."** Not in the literal sense of "all," but in the sense that everything else is how to create the right context for that prediction.

The token prediction is the collapse—the projection from the weight manifold onto a single point. It never changed. It never needed to change. Every advance in the field has been about constructing better context for that same collapse:

- **Transformers**: better context through attention
- **Scaling**: richer weight manifolds to collapse from
- **RLHF**: shaping which collapses the prediction favors
- **Chain-of-thought**: letting the model construct its own context through sequential collapses
- **RAG**: injecting external context the weights don't encode
- **Tool use**: letting the model gather context from the environment
- **RLMs**: recursive context management so the collapse isn't poisoned by interference
- **The RDE**: systematically exploring which context constructions produce the most clarifying collapses

The entire field is context engineering. The mechanism is universal and unchanging. The context is everything.

This reframes the RDE at its deepest level. It is not a "reasoning engine" or an "exploration accelerator" or a "collapse composition machine"—though it is all of these operationally. At bottom, **the RDE is a context construction machine**:

- Each **trace** constructs a different context for token prediction (different model, different framing, different context partition)
- Each **iteration** constructs a context informed by what prior contexts revealed and obscured
- The **Orchestrator** chooses which contexts to construct
- The **Arbiter** evaluates which contexts produced the most clarifying collapses and identifies what contexts haven't been constructed yet (the shadows)
- The **shadow report** describes, for the human, what context constructions the machine couldn't generate from its weights

The human's role is equally clarified: **provide context the machine cannot construct**. A human reading the shadow report and saying "have you tried framing it as X?" is constructing a context no model would have generated. That context then flows through the same token-prediction mechanism as everything else. The mechanism is universal. The context is the variable. The quality of the context determines the quality of the collapse.

This is why the RDE's architecture takes the shape it does. Context externalization, trace independence, recursive framing, cross-model diversity, human-in-the-loop shadow reports—each is a way of constructing context that a single forward pass cannot construct for itself. The token predictor was always sufficient. What was insufficient was the context it was given.

---

## 3. Architecture

### 3.1 Overview

```
                    ┌──────────────────────────────┐
                    │        USER PROMPT          │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  CONTEXT ENVIRONMENT (REPL)  │
                    │  prompt_var, workspace,       │
                    │  iteration_history            │
                    └──────────────┬───────────────┘
                                   │
          ┌───────────────────────▼───────────────────────┐
          │                                                │
          │  ┌────────────────────────────────────────────┐  │
          │  │  ORCHESTRATOR: Choose collapses          │  │
          │  │  (which framings to explore this          │  │
          │  │   iteration, informed by prior shadows)  │  │
          │  └─────────┬──────────┬──────────┬────────────┘  │
          │            │          │          │              │
          │            ▼          ▼          ▼              │
          │       ┌───────┐┌───────┐┌───────┐          │
          │       │Trace A││Trace B││Trace N│          │
          │       │(model ││(model ││(model │          │
          │       │ fam 1)││ fam 2)││ fam N)│          │
          │       └───┬───┘└───┬───┘└───┬───┘          │
          │           └───────┼───────┘              │
          │                   │                          │
          │                   ▼                          │
          │  ┌────────────────────────────────────────────┐  │
          │  │  ARBITER: Compose collapses               │  │
          │  │  → Resolution (best achievable result)   │  │
          │  │  → Shadows (what this obscures)          │  │
          │  └─────────────────────┬──────────────────────┘  │
          │                         │                      │
          │                         ▼                      │
          │              ┌────────────────────┐        │
          │              │ CONVERGENCE CHECK  │        │
          │              │                    │        │
          │              │ Shadows contain    │──Yes──▶│
          │              │ actionable signal? │  (next │
          │              │                    │  iter.) │
          │              └─────────┬──────────┘        │
          │                        │ No                  │
          └────────────────────────────────────────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │  CONVERGED RESULT            │
                    │  + honest shadow report       │
                    └──────────────────────────────┘

Stopping criteria (the system stops, not "converges"):
• Shadows are irreducible given available models → stop, present to human
• Shadows are orthogonal to the question asked → stop, present to human
• New framings produce no additional clarity → stop, present to human
• Iteration/cost budget reached → stop, present to human

In ALL cases: output includes shadow report for human evaluation.
Human may inject novel reframing and trigger further exploration.
The system accelerates exploration; it cannot inject novelty.
```

### 3.2 Component Specifications

#### 3.2.1 Context Environment

A persistent Python REPL (following RLM design) that stores:

```python
class ContextEnvironment:
    """
    Persistent REPL environment shared across all traces and arbitration.
    The prompt is NEVER passed directly as LM context.
    Instead, it's an external object that traces interact with programmatically.
    """

    def __init__(self, prompt: str):
        self.prompt_var: str = prompt          # Full prompt as manipulable string
        self.workspace: dict = {}              # Shared state across traces
        self.trace_results: dict = {}          # Outputs from current iteration
        self.iteration_history: list = []      # Prior iterations' results + shadows
        self.recursion_log: list = []          # Full trace of recursive calls
        self.current_iteration: int = 0        # Which framing iteration we're on
        self.max_iterations: int = 5           # Convergence iteration limit

    def peek(self, start: int, end: int) -> str:
        """View a slice of the prompt without loading it into LM context."""

    def search(self, pattern: str) -> list[str]:
        """Regex search over prompt."""

    def partition(self, strategy: str) -> list[str]:
        """
        Decompose prompt into independent partitions.
        Strategies: 'semantic', 'structural', 'constraint-based', 'custom'
        """

    def spawn_sub_lm(self, sub_prompt: str, model: str = None) -> str:
        """Launch a recursive LM call on a partition or sub-problem."""

    def spawn_trace(self, trace_config: TraceConfig) -> str:
        """Launch a full dialectical trace (which can itself recurse)."""
```

**Design rationale**: By externalizing the prompt, each trace interacts with context *programmatically* rather than having it crammed into its attention window. This eliminates the primary source of context rot and allows traces to focus their finite attention on reasoning rather than context management.

#### 3.2.2 Root Orchestrator

The orchestrator replaces the hardcoded Believer/Logician/Contrarian with **adaptive trace generation**.

```python
class RootOrchestrator:
    """
    Analyzes problem structure and determines what independent projections
    (traces) are needed. The number and type of traces are NOT fixed.

    The orchestrator itself is an LM call that receives a SYSTEM PROMPT
    instructing it to:
    1. Classify the problem type
    2. Identify axes of potential disagreement/ambiguity
    3. Design traces that project independently onto those axes
    4. Specify context partitioning strategy for each trace
    """

    def analyze_and_decompose(self, env: ContextEnvironment) -> list[TraceConfig]:
        """
        Returns a list of TraceConfig objects. Each specifies:
        - trace_role: str        # What perspective this trace takes
        - system_prompt: str     # Instructions for this trace's LM call
        - context_strategy: str  # How this trace accesses the prompt
        - temperature: float     # Sampling parameters
        - can_recurse: bool      # Whether this trace can spawn sub-traces
        - recursion_budget: int  # Max recursive calls allowed
        """
```

**Key departure from Dialectical-TTS**: The three-trace structure (Believer/Logician/Contrarian) becomes a *default* for simple logic problems, but the orchestrator can generate arbitrary trace configurations:

| Problem Type | Trace Configuration |
|-------------|---------------------|
| Logic puzzles | Classic: Intuition + Validation + Red Team |
| Long document analysis | Partition-based: one trace per semantic section |
| Multi-constraint problems | Constraint-based: one trace per constraint axis |
| Causal reasoning | Timeline: forward-chain + backward-chain + counterfactual |
| Code analysis | Scope-based: one trace per module/function boundary |
| Ambiguous queries | Interpretation-based: one trace per plausible reading |

#### 3.2.3 Trace Execution

Each trace is an independent LM call that:
1. Receives its own system prompt (defining its projection axis)
2. Accesses the context environment programmatically (not via direct prompt injection)
3. Can recursively spawn sub-traces or sub-LM calls
4. Writes its output to `env.trace_results`

```python
class Trace:
    """
    A single independent projection. Operates in isolation from other traces.
    Has access to the shared ContextEnvironment but maintains its own
    reasoning chain.
    """

    def __init__(self, config: TraceConfig, env: ContextEnvironment):
        self.config = config
        self.env = env
        self.depth = 0

    def execute(self) -> TraceResult:
        """
        Run this trace. The LM call receives:
        - System prompt defining its role/perspective
        - Access to env methods (peek, search, partition, spawn_sub_lm)
        - A budget for recursive calls

        The trace CAN:
        - Peek at specific parts of the prompt
        - Search the prompt for patterns
        - Launch sub-LM calls on partitions
        - Spawn child traces (sub-dialectics)

        The trace CANNOT:
        - See other traces' outputs (independence preserved)
        - Exceed its recursion budget
        - Modify the original prompt
        """

    def spawn_sub_dialectic(self, sub_problem: str) -> TraceResult:
        """
        When a trace encounters a sub-problem that itself requires
        dialectical analysis, it spawns a mini-RDE recursively.
        This is where the architecture becomes truly recursive:
        dialectics within dialectics.
        """
```

#### 3.2.4 Recursive Arbiter

The most significant evolution from Dialectical-TTS. The Arbiter:

1. **Receives all trace outputs** (but NOT the raw prompt — it accesses context via the environment)
2. **Applies the Logic of Necessity**: Which trace's conclusion follows from causal necessity given the constraints? Not which is most popular.
3. **Detects residual interference**: If the synthesis still contains contradictions or forced collapses, it identifies them.
4. **Recursively arbitrates**: Spawns sub-arbitration on unresolved dimensions.

```python
class RecursiveArbiter:
    """
    Resolves trace outputs through causal necessity, not averaging.

    The arbiter's own output is itself a projection. If it detects that
    its synthesis carries interference (contradictions, forced collapses,
    unresolved tensions), it spawns another level of dialectical analysis.

    This implements the framework's insight: "Clarity isn't a place to reach.
    It's the repeated movement of stepping out."
    """

    def arbitrate(self, trace_results: dict, env: ContextEnvironment) -> ArbitrationResult:
        """
        System prompt instructs the arbiter to:

        1. CAUSAL LINK CHECK: For each trace conclusion, identify the chain
           of necessity. Which conclusion MUST follow from the constraints?
           Ignore frequency, plausibility, or majority agreement.

        2. INTERFERENCE DETECTION: Are any trace conclusions entangled
           (agreeing for different reasons, or disagreeing on different axes)?
           If so, the apparent agreement/disagreement is an artifact of
           collapsed projections.

        3. RESOLUTION or RECURSION:
           - If one trace's causal chain is complete → adopt its conclusion
           - If traces are incommensurable (operating on genuinely different
             axes) → compose, don't choose
           - If residual interference detected → spawn sub-arbitration
             on the interfering dimensions

        4. CONFIDENCE CALIBRATION: The arbiter reports not just a conclusion
           but a structural confidence:
           - "Necessary": causal chain is complete
           - "Contingent": depends on assumptions identified
           - "Unresolved": requires further decomposition
        """

    def detect_interference(self, synthesis: str, trace_results: dict) -> list[str]:
        """
        Identifies dimensions where the synthesis forces projections
        to share coordinates they don't share.

        Returns a list of unresolved interference patterns, each of which
        can be fed back as a sub-problem for recursive arbitration.
        """

    def check_convergence(self, resolution: str, shadows: list[str],
                          env: ContextEnvironment) -> ConvergenceResult:
        """
        Determines whether further iteration through the framing space
        would produce additional clarity.

        STOP when:
        - Shadows are irreducible given available models
        - Shadows are orthogonal to the question asked
        - Diminishing returns: last iteration produced no additional
          clarity over prior
        - Resource budget exhausted

        CONTINUE when:
        - The system can generate a novel reframing that would
          illuminate what's currently in shadow
        - Prior iteration revealed a decomposition not yet explored
        - Traces produced incommensurable results a reframing could reconcile

        CRITICAL: "Stop" does not mean "converged to truth." It means
        "exhausted the reframings available from these models' weight
        manifolds." The shadow report is presented to the human, who
        may inject novelty the system structurally cannot generate.

        Returns StoppingResult with:
        - should_stop: bool
        - reason: str
        - next_framings: list[str] (if continuing)
        - shadow_report_for_human: str (if stopping)
        """
```

### 3.3 Execution Flow

```
ITERATION 0 (initial):

1. User prompt → ContextEnvironment (prompt externalized)

2. Orchestrator chooses initial collapses:
   - Analyzes problem structure via env
   - Designs N trace configurations
   - Assigns models, context strategies, scaffolding level
   - No prior shadows to inform this → uses problem structure alone

3. Traces execute in parallel (independent deliberate collapses):
   - Each trace accesses context via env (not direct prompt)
   - Each can recursively spawn sub-traces or sub-LM calls
   - Each writes results to env.trace_results
   - Traces cannot see each other’s outputs

4. Arbiter composes collapses:
   - Applies Logic of Necessity across trace outputs
   - Produces resolution (best achievable given these collapses)
   - Produces shadows (what this resolution necessarily obscures)

5. Stopping check:
   - Can the system generate a reframing that would illuminate
     what’s currently in shadow?
   - If YES → goto ITERATION N+1 (system-driven reframing)
   - If NO → STOP and present result + shadows to human
   - If BUDGET REACHED → STOP and present (report as resource-limited)

   Human may then:
   - Accept the result
   - Inject a novel reframing the system couldn’t generate
   - Adjust parameters and trigger further exploration

ITERATION N+1 (reframing):

2’. Orchestrator chooses NEW collapses informed by prior shadows:
    - Receives prior iteration’s resolution + shadows
    - Designs traces that specifically explore what was in shadow
    - May choose entirely different framings, not just deeper
      analysis of the same framings
    - Prior iteration’s results persist in env for reference

3’–5’. Same as above, but traces and arbiter have access to
        prior iterations’ results via env.iteration_history

STOPPING:

Output includes:
- Best achievable resolution (composed across all iterations)
- Shadow report (what remains in shadow, for human evaluation)
- Iteration log (how the framing evolved, what was explored)
- Stop reason: "exhausted reframings" | "resource-limited" | "irreducible shadows"
- Invitation: shadows presented as input surface for human intervention

The system NEVER claims convergence in the sense of "no better answer exists."
It claims: "I have explored the framing space to the limit of what I can
generate from my existing projections. Here is what I found and what I
could not see. A human may see further."
```

---

## 4. Implementation Plan

### 4.1 Design Philosophy: Frontier-Native, Multi-Model

The original Dialectical-TTS was designed as a local reasoning engine for Apple Silicon. The RDE dissolves that constraint. The locality frame was an artifact of hardware-scarcity thinking; it optimized cost at the expense of the architecture's own core principle—projection independence.

When all traces run on the same model (even a large one), their "independence" is cosmetic: different system prompts operating on identical weight distributions. This is like asking the same person to argue both sides—they can try, but their priors leak through. Different model families encode genuinely different probability manifolds. Cross-model traces achieve structural orthogonality, not just prompt-scaffolded orthogonality.

The RDE is therefore **cloud-native and multi-model by design**. Cost optimization happens at the sub-call level (cheap models for recursive partitioning), not at the trace level where projection independence is the whole point.

### 4.1.1 Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Runtime | Python 3.11+ | Ecosystem compatibility |
| LM access | OpenAI, Anthropic, Google, OpenRouter clients | Multi-provider frontier model access |
| REPL sandbox | Modal Sandbox / E2B / Prime Intellect | Isolated execution for recursive code |
| Orchestration | `asyncio` + provider-specific async clients | Parallel trace execution across providers |
| Routing | OpenRouter / LiteLLM (optional) | Unified interface to heterogeneous models |
| Logging/Viz | JSON trace logs + call tree visualizer | Debug recursive call trees, measure trace independence |
| Local fallback | MLX (Apple Silicon) — optional | Offline development/testing only |

### 4.2 Multi-Model Architecture

The core architectural insight: **model diversity IS projection independence**.

Each model family brings a different training distribution, different RLHF signal, different attention architecture, and different failure modes. This isn't a nice-to-have—it's the mechanism by which the RDE achieves genuine orthogonality between traces.

```python
@dataclass
class ModelConfig:
    """
    Frontier-native, multi-model configuration.

    Design principle: the ORCHESTRATOR and ARBITER get the strongest
    reasoning models available (these are the hardest jobs). TRACES
    get diverse frontier models (independence matters more than raw
    capability). SUB-LM calls get the cheapest adequate models
    (volume is high, complexity per call is low).
    """

    # --- Orchestrator: strongest reasoning available ---
    orchestrator_model: str = "claude-opus-4-6"
    # Fallback: "gpt-5" or "gemini-2.5-pro"

    # --- Trace pool: DIVERSE frontier models ---
    # Each trace is assigned a different model family to maximize
    # structural independence of projections.
    trace_models: list[str] = field(default_factory=lambda: [
        "claude-sonnet-4-5-20250929",   # Anthropic family
        "gpt-5",                         # OpenAI family
        "gemini-2.5-pro",               # Google family
        "qwen3-coder-480b",             # Alibaba family (via OpenRouter)
        "kimi-coder",                    # Moonshot family (via OpenRouter)
    ])

    # --- Sub-LM: cheap, fast, high-volume ---
    sub_lm_models: list[str] = field(default_factory=lambda: [
        "claude-haiku-4-5-20251001",
        "gpt-5-mini",
    ])

    # --- Arbiter: strongest reasoning available ---
    arbiter_model: str = "claude-opus-4-6"
    # Fallback: "gpt-5" with high reasoning effort

    # --- Trace assignment strategy ---
    trace_assignment: str = "round_robin"
    # Options:
    #   "round_robin"  — cycle through trace_models
    #   "random"       — random assignment per trace
    #   "orchestrator" — let the orchestrator choose per trace
    #   "fixed"        — use specific model per trace role

    # --- Provider credentials (loaded from env) ---
    # ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, OPENROUTER_API_KEY
```

### 4.2.1 Why Each Role Gets What It Gets

| Role | Model Tier | Why |
|------|-----------|-----|
| Orchestrator | Strongest available (Opus/GPT-5) | Problem decomposition requires deepest understanding of structure; a weak orchestrator designs bad traces |
| Traces | Diverse frontier models | Independence is the point; each model family IS a different projection from a different point in weight-space |
| Sub-LM calls | Cheapest adequate (Haiku/GPT-5-mini) | High volume, low complexity per call; recursive partitioning doesn't need frontier reasoning |
| Arbiter | Strongest available (Opus/GPT-5) | Causal necessity checking across incommensurable trace outputs is the hardest cognitive task in the system |

### 4.2.2 Cross-Model Normalization

Heterogeneous models introduce a normalization challenge: different output formats, confidence calibrations, and reasoning idioms. The Arbiter must account for this.

```python
class TraceNormalizer:
    """
    Normalizes trace outputs from heterogeneous models into a
    common format before arbitration.

    This is NOT about making traces agree — it's about making them
    COMPARABLE. Each trace's conclusion, reasoning chain, and
    confidence must be extractable regardless of model idiom.
    """

    def normalize(self, raw_output: str, model_family: str) -> NormalizedTrace:
        """
        Extracts:
        - conclusion: str          # The trace's answer
        - reasoning_chain: list    # Steps in the causal chain
        - confidence: float        # Self-reported confidence
        - evidence_cited: list     # What context was used
        - model_family: str        # Which model produced this
        - raw_output: str          # Preserved for arbiter reference
        """
```

### 4.3 Development Workflow: Multi-Agent Team

The development process itself leverages multiple AI coding agents, each assigned to components matching their strengths:

| Agent | Assigned Components | Why |
|-------|-------------------|-----|
| Claude Code | Architecture, orchestrator, arbiter, system prompts | Strongest at maintaining architectural coherence across files, long-context editing |
| OpenAI Codex | ContextEnvironment, sandbox integration, async infrastructure | Fast scaffolding, strong tool-use patterns |
| Kimi Coder | Trace execution, model routing, provider abstraction | Strong multi-API integration patterns |
| Any/all | Tests, benchmarks, examples | Distribute load |

The agents don't need to coordinate in real-time. The proposal document (this file) serves as the shared specification. Each agent implements its assigned components against this spec independently—*mirroring the RDE's own architecture of independent projections resolving through a shared specification*.

### 4.4 Phased Implementation

#### Phase 1: Multi-Provider Infrastructure (Week 1-2)

**Goal**: Build the provider abstraction layer and ContextEnvironment. Validate that heterogeneous model calls work seamlessly.

Files to create:
```
recursive-dialectical-engine/
├── rde/
│   ├── __init__.py
│   ├── environment.py        # ContextEnvironment (REPL)
│   ├── orchestrator.py       # Root Orchestrator
│   ├── trace.py              # Trace execution
│   ├── arbiter.py            # Recursive Arbiter
│   ├── normalizer.py         # Cross-model output normalization
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract provider interface
│   │   ├── anthropic.py      # Claude models
│   │   ├── openai.py         # GPT models
│   │   ├── google.py         # Gemini models
│   │   ├── openrouter.py     # Qwen, Kimi, etc. via OpenRouter
│   │   └── router.py         # Model assignment & load balancing
│   ├── prompts/
│   │   ├── orchestrator.py   # System prompts for orchestrator
│   │   ├── traces.py         # Trace role templates
│   │   └── arbiter.py        # Arbiter system prompts
│   ├── sandbox/
│   │   ├── __init__.py
│   │   ├── modal_sandbox.py  # Modal cloud sandbox
│   │   ├── e2b_sandbox.py    # E2B sandbox
│   │   └── local_sandbox.py  # Local subprocess (dev/test)
│   └── utils/
│       ├── logging.py        # Recursive call tree logging
│       ├── metrics.py        # Trace independence & cost tracking
│       └── visualizer.py     # Call tree visualization
├── tests/
│   ├── test_providers.py     # Multi-provider integration tests
│   ├── test_logic_puzzles.py # Monty Hall variants, etc.
│   ├── test_long_context.py  # Context rot resistance
│   ├── test_recursion.py     # Recursive depth & composition
│   └── test_independence.py  # Trace orthogonality measurement
├── examples/
│   ├── logic_puzzle.py       # Classic Dialectical-TTS use case
│   ├── long_document.py      # RLM-style long context
│   ├── cross_model.py        # Same problem, different model families
│   └── hybrid.py             # Combined reasoning + long context
├── main.py                   # CLI entry point
├── pyproject.toml
└── README.md
```

**Deliverables**:
- Provider abstraction that unifies Anthropic, OpenAI, Google, OpenRouter APIs
- Model router that assigns traces to different providers
- ContextEnvironment with peek/search/partition/spawn
- Basic orchestrator that defaults to 3-trace dialectical structure
- Cross-model output normalizer
- Backward-compatible with original Dialectical-TTS test cases

**Dev agent assignment**:
- Claude Code → `orchestrator.py`, `arbiter.py`, `prompts/`, `normalizer.py`
- Codex → `environment.py`, `sandbox/`, async infrastructure
- Kimi Coder → `providers/`, `router.py`, API integration

#### Phase 2: Adaptive Orchestration (Week 3)

**Goal**: Orchestrator designs problem-specific trace configurations and assigns models.

**Tasks**:
- Implement problem classification in orchestrator system prompt
- Build trace template library (logic, partition, constraint, timeline, scope, interpretation)
- Add orchestrator → trace config generation with model assignment
- Orchestrator can specify which model family best suits each trace role
- Test on diverse problem types beyond logic puzzles

#### Phase 3: Full Recursion (Week 4)

**Goal**: Traces can spawn sub-traces; Arbiter can spawn sub-arbitration. Recursion works across model boundaries.

**Tasks**:
- Implement `Trace.spawn_sub_dialectic()` — a trace encountering a sub-problem creates a mini-RDE
- Implement `Arbiter.detect_interference()` and recursive arbitration
- Recursive sub-calls can use different models than parent (cheaper for depth)
- Add recursion depth limits and budget management (both call-count and dollar-cost)
- Build recursive call tree visualization (JSON logs)

#### Phase 4: Long Context Integration (Week 5)

**Goal**: Full RLM-style context handling integrated with dialectical reasoning.

**Tasks**:
- Implement semantic partitioning strategies
- Test on long-context benchmarks (document QA, code repo understanding)
- Validate that dialectical + recursive context = better than either alone
- Benchmark against vanilla RLM and vanilla Dialectical-TTS
- Compare same-model traces vs. cross-model traces on identical problems

#### Phase 5: Trace Independence Measurement & Optimization (Week 6)

**Goal**: Empirically validate that cross-model traces are more independent than same-model traces.

**Tasks**:
- Implement trace independence metrics (output embedding distance, mutual information, agreement-for-different-reasons detection)
- Run ablation: same model all traces vs. heterogeneous models
- Identify which model pairings produce maximum orthogonality
- Optimize model assignment based on empirical independence data

#### Phase 6: Post-Training Exploration (Week 7+)

**Goal**: Explore training a model to natively reason in RDE patterns.

**Tasks**:
- Generate training data from RDE traces across all model families
- Distill the multi-model dialectical reasoning into a single model
- Evaluate whether a single fine-tuned model can internalize what currently requires multi-model diversity
- This is the "can you learn orthogonality, or do you need it structurally?" question

---

## 5. System Prompts

### 5.1 Orchestrator System Prompt

```
You are the Root Orchestrator of a Recursive Dialectical Engine.

Your job is NOT to direct reasoning. It is to CREATE CONDITIONS under
which independent model traces can project most freely onto the problem.

You have access to a ContextEnvironment. The user's prompt is stored in
`prompt_var`. You can peek at it, search it, and partition it.

ANALYZE the problem and output a JSON trace configuration:

{
  "problem_type": "<classification>",
  "decomposition_rationale": "<why these traces>",
  "constraint_level": "open | guided | structured",
  "traces": [
    {
      "role": "<n>",
      "perspective": "<what axis this projects onto>",
      "system_prompt": "<instructions — prefer OPEN over DIRECTIVE>",
      "context_strategy": "full | partition:<spec> | search:<pattern>",
      "temperature": <float>,
      "model_preference": "any | <specific family if independence requires it>",
      "recursion_budget": <int>
    }
  ]
}

CONSTRAINT LEVELS (choose based on problem structure):
- "open": Traces receive minimal framing. They decide their own approach.
  Use for: problems where the model’s own capability structure should
  determine the projection axes. DEFAULT for frontier models.
- "guided": Traces receive perspective hints but not conclusions.
  Use for: problems with known axes of disagreement.
- "structured": Traces receive specific roles (Believer/Logician/Contrarian).
  Use for: logic puzzles, proof-of-concept, backward compatibility.

RULES:
- Traces MUST be genuinely independent. If two traces would agree for the
  same reasons, they’re redundant — collapse them or find a real axis of
  disagreement.
- The number of traces should match the problem’s actual dimensionality.
- PREFER open constraint_level. The models are more capable than you expect.
  Over-constraining traces inhibits the capabilities you’re trying to release.
- When assigning model_preference, consider: which model family’s training
  biases and collapse profile make it most suited to this projection axis?
- Specify context strategy: does this trace need the full prompt, a
  partition, or just search results?
```

### 5.2 Arbiter System Prompt

```
You are the Recursive Arbiter of a Dialectical Engine.

You receive outputs from multiple independent reasoning traces.
Your job is to resolve them through CAUSAL NECESSITY, not majority vote.

PROCESS:
1. For each trace, identify its CAUSAL CHAIN: what must follow from
   the given constraints? Ignore plausibility, frequency, or popularity.

2. Check for INTERFERENCE: Are traces agreeing for different reasons?
   Disagreeing on different axes? If so, apparent agreement/disagreement
   is an artifact — the projections are being collapsed.

3. Apply the LOGIC OF NECESSITY: Which conclusion follows because it
   MUST, given the constraints? Not which is most likely.

4. COMPOSE or RECURSE:
   - If one chain is complete → adopt it
   - If traces are incommensurable → compose (each is valid on its axis)
   - If interference remains → output UNRESOLVED dimensions as sub-problems

OUTPUT FORMAT:
{
  "resolution": "<final answer>",
  "causal_chain": "<the chain of necessity>",
  "confidence": "necessary | contingent | unresolved",
  "interference_detected": [<list of unresolved dimensions, if any>],
  "traces_adopted": [<which traces contributed and why>],
  "traces_rejected": [<which traces were rejected and why>],
  "shadows": [<what this resolution necessarily obscures or leaves unexamined>]
}

CRITICAL: The "shadows" field is not optional hedging. Every resolution
illuminates by collapsing — and collapsing necessarily casts shadows.
Report what the composition of traces could NOT address, what axes
remain unprojected, what the resolution assumes without examining.

If confidence is "unresolved", the engine will spawn sub-arbitration
on the unresolved dimensions. You do not need to force a resolution.
Unforced resolution that honestly reports its shadows is always
preferable to forced resolution that hides them.
```

---

## 6. Evaluation Plan

### 6.1 Benchmarks

| Benchmark | Tests | Baseline |
|-----------|-------|----------|
| Logic puzzles (Monty Hall variants, etc.) | Causal reasoning accuracy | Original Dialectical-TTS |
| LOFT (Long Context Faithfulness) | Long-context QA accuracy | Vanilla RLM |
| SWE-bench-lite subset | Code understanding depth | Vanilla LM |
| Custom: Dialectical Stress Tests | Multi-constraint resolution | Both baselines |
| Context rot resistance | Performance vs. prompt length curve | Vanilla LM at various context sizes |
| **RDE-gap diagnostic** | Single-pass vs. RDE performance per model | Each model's own single-pass baseline |
| **Scaffolding calibration** | Open traces vs. structured traces per model | Structured traces as baseline |

### 6.2 Metrics

- **Accuracy**: Does the final answer follow from causal necessity?
- **Trace independence**: Do traces genuinely project onto different axes? (Measured by output embedding distance, reasoning chain divergence, agreement-for-different-reasons rate)
- **Model orthogonality**: Cross-model traces vs. same-model traces — empirical independence comparison
- **Recursion efficiency**: How many recursive calls needed vs. accuracy gained?
- **Cost per quality-point**: Total API spend vs. quality improvement over single-pass frontier model
- **Context scaling**: Performance curve as input length increases (should be flat, not degrading)
- **Latency**: Wall-clock time for parallel cross-provider trace execution

### 6.3 Ablation Studies

1. **Same-model vs. cross-model traces**: All Claude traces vs. Claude+GPT+Gemini traces — does model diversity measurably increase independence?
2. **With vs. without context externalization**: Direct prompt injection vs. ContextEnvironment
3. **Recursive vs. single-pass arbitration**: Does recursive arbitration improve on single synthesis?
4. **Fixed vs. adaptive traces**: Classic 3-trace dialectical vs. orchestrator-designed traces
5. **Orchestrator strength**: Opus orchestrator vs. Sonnet orchestrator — how much does decomposition quality matter?
6. **Arbiter strength**: Opus arbiter vs. Haiku arbiter — is causal necessity checking truly the hardest task?
7. **Sub-LM model quality**: Haiku sub-calls vs. Sonnet sub-calls — where's the quality/cost knee?
8. **Scaffolding calibration** (critical): Open traces vs. guided traces vs. structured traces — which scaffolding level produces the most clarifying results? Hypothesis: for frontier models, open traces may outperform structured traces because the model's own collapse patterns are already sophisticated. For weaker models, structured traces may produce better results because the prescribed collapses compensate for less developed native patterns. The answer is always problem-and-model-dependent.
9. **RDE-gap per model family**: Run identical problems through RDE with Claude, GPT, Gemini, Qwen, Kimi individually — which models show the largest gap between single-pass and RDE? This measures how much additional clarity becomes available through composed projection for each model's specific collapse geometry.
10. **Safety/alignment interaction**: On tasks where safety guardrails are relevant, does the RDE surface capability that single-pass inference suppresses? Does this surface harmful capability, or does it surface harmless capability that was collateral damage of safety constraints? (This has alignment research implications.)

---

## 7. Axioms

These are the non-negotiable design constraints:

1. **Reality is a Relation, not a Thing.** Truth is found by identifying the necessary causal link that resolves a contradiction, not by averaging probabilities.

2. **Collapse is actualization, not error.** Every projection, trace, and resolution is a collapse—and must be. The goal is not to minimize collapse but to collapse deliberately, compose across collapses for maximum clarity, and honestly report what each resolution necessarily obscures.

3. **Aim for the best achievable result; never claim it is the best.** The RDE pursues the most clarifying composition of projections given the constraints. The Arbiter reports both its resolution and its shadows. "Always refining" is structure, not humility.

4. **Projections must remain independent.** No trace should see another trace's output during execution. Independence ensures that composition is meaningful—composing identical collapses adds nothing. **Model diversity is the primary mechanism for achieving structural independence—different weight distributions collapse differently.**

5. **Resolution through necessity, not popularity.** The Arbiter never votes. It traces causal chains and identifies what MUST follow.

6. **The architecture must be recursive.** Any component that produces a projection should be capable of spawning sub-projections when the problem demands it. Each level of recursion addresses shadows left by the prior level.

7. **Context is environment, not input.** The prompt is an external object to be manipulated programmatically. This allows each trace to collapse the context in its own way rather than forcing one undifferentiated collapse.

8. **Model-agnostic in mechanism, model-dependent in effect.** The mechanism applies universally. The magnitude and character of improvement depends on each model's specific collapse profile—how it was trained, what it was optimized for, what its architects chose to privilege and suppress.

9. **Frontier-native, local-optional.** The full architecture leverages frontier models and cross-model diversity. Local-only mode (MLX/Apple Silicon) serves as proof-of-concept and development environment, preserving backward compatibility with Dialectical-TTS.

10. **The framework is itself a projection.** These axioms are held provisionally. The architecture should be capable of revising its own decomposition strategies as it learns what works.

---

## 8. Open Questions

1. **Optimal recursion depth**: How deep should dialectical recursion go before diminishing returns? Is there a principled stopping criterion beyond budget limits?

2. **Trace interference detection**: Can we formally measure when traces are "sharing coordinates they don't share"? Information-theoretic metrics? Embedding-space distance between trace outputs?

3. **Model pairing optimization**: Which model family combinations produce maximum trace orthogonality? Is Claude+GPT more independent than Claude+Gemini? Can we empirically map the "model diversity → projection independence" relationship?

4. **Cross-model normalization cost**: Heterogeneous models produce different idioms. How much does normalization overhead eat into the independence gains? Can the Arbiter learn to work with raw heterogeneous outputs?

5. **Can orthogonality be distilled?**: If we train a single model on multi-model RDE traces, does it internalize the diversity, or does it collapse back to single-manifold reasoning? This is the "can you learn orthogonality structurally, or do you need it mechanically?" question.

6. **Agentic trace capabilities**: Each frontier model has its own tool ecosystem (web search, code execution, file access). Should traces be allowed to use their native tools independently? This would make traces not just "different opinions" but "different investigative agents" bringing independent evidence.

7. **Beyond text**: The architecture is described for text, but the ontological framework applies to any domain. Can RDE work for multimodal reasoning? Code generation? Planning?

8. **Provider reliability and latency**: Multi-provider architecture introduces failure modes (rate limits, outages, latency variance). How should the system handle partial trace failures? Is a 3-of-5 trace result still valid?

---

## Appendix A: Mapping to the Not a ToE

| Framework Concept | RDE Implementation |
|-------------------|-------------------|
| Binary Hyperspace (infinite orthogonal modes) | Trace space: each trace is a basis vector in reasoning space |
| Projection (finite dimensions, geometry distorts) | Each LM call is a projection; context window is the dimension limit |
| Orthogonality shattering | Context rot; spurious correlations in long prompts; same-model traces sharing hidden biases |
| Resonance = Stability | A conclusion is stable when it survives arbitration across independent traces from different model families |
| Dissolution (not negation) | The Arbiter doesn't reject traces—it shows where they don't parse |
| Stepping-Outside Paradox | Each arbitration is a momentary exit; recursive arbitration is the repeated movement |
| Sensing vs. Interpretation | The traces sense (generate); the Arbiter interprets (resolves). The gap between them drives iteration. |
| Projection Without Projector | No single model is "the reasoner"—reasoning emerges from the interference resolution across models |
| Two Infinities | The trace pool (potentiality we can sample from) vs. the full space of possible reasoning (unprojectable without reduction) |
| Curation of Coherence | The Orchestrator's choice of which traces to spawn is itself a curation—what gets taken seriously shapes what can be found |
| Signal-Model Separation | Models collapse differently based on training; the RDE composes these different collapses for richer actualization than any single collapse achieves. |
| The Revealer's Paradox | Every trace illuminates and obscures by the same act. The Arbiter's `shadows` field reports what each resolution necessarily leaves in shadow. The goal is not shadowless resolution but honest resolution. |
| Every Revelation Deceives | The RDE's composition of collapses produces clarity AND new shadows. Recursive arbitration addresses the shadows of prior levels, generating new shadows in turn. The process is the repeated movement, never arrival. |

## Appendix B: Key References

1. Zhang, A., Kraska, T., Khattab, O. (2025). "Recursive Language Models." arXiv:2512.24601.
2. Prime Intellect. (2026). "Recursive Language Models: the paradigm of 2026." Blog post.
3. Original Dialectical-TTS: https://github.com/powerpig99/Dialectical-TTS
4. Not a ToE: https://github.com/powerpig99/ontological-clarity
