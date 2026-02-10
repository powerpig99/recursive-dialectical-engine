# API Reference

## Core Engine

### `DialecticalEngine` (`rde/engine.py`)

Main orchestration loop. Coordinates all components.

```python
class DialecticalEngine:
    def __init__(self, config: ModelConfig | None = None) -> None
    async def __aenter__(self) -> DialecticalEngine
    async def __aexit__(self, *exc) -> None
    async def run(
        self,
        prompt: str,
        use_orchestrator: bool = True,
        max_iterations: int = 1,
        budget: RecursionBudget | None = None,
    ) -> EngineResult
```

**Usage:**
```python
async with DialecticalEngine(ModelConfig()) as engine:
    result = await engine.run("Your problem here", max_iterations=3)
```

---

### `ContextEnvironment` (`rde/environment.py`)

Externalized prompt REPL. The prompt is never passed directly as LLM context.

```python
class ContextEnvironment:
    def __init__(
        self,
        prompt: str,
        max_iterations: int = 5,
        router: ModelRouter | None = None,
        budget: RecursionBudget | None = None,
        sub_lm_models: list[str] | None = None,
        sub_lm_temperature: float = 0.3,
    )

    def peek(self, start: int, end: int) -> str
    def search(self, pattern: str) -> list[str]
    def partition(self, strategy: str = "structural") -> list[str]
    async def prepare_partitions(self, strategy: str, custom_fn=None) -> list[str]
    async def spawn_sub_lm(self, sub_prompt: str, model: str | None = None) -> str
    def store_iteration(self, results: dict, arbitration: dict) -> None
```

---

### `Orchestrator` (`rde/orchestrator.py`)

Designs trace configurations via LLM analysis.

```python
class Orchestrator:
    def __init__(self, router: ModelRouter, model: str, temperature: float = 0.3) -> None

    async def design_traces(self, env: ContextEnvironment) -> list[TraceConfig]
    async def design_traces_for_iteration(
        self, env: ContextEnvironment,
        prior_arbitration: ArbitrationResult,
        iteration: int,
    ) -> list[TraceConfig]
```

---

### `TraceExecutor` (`rde/trace.py`)

Executes individual traces via the model router.

```python
class TraceExecutor:
    def __init__(
        self,
        router: ModelRouter,
        environment: ContextEnvironment,
        budget: RecursionBudget | None = None,
    ) -> None

    async def execute(self, config: TraceConfig, model: str) -> TraceResult
    async def spawn_sub_dialectic(
        self, sub_prompt: str, budget: RecursionBudget | None = None,
    ) -> EngineResult
```

---

### `TraceNormalizer` (`rde/normalizer.py`)

Normalizes trace outputs for cross-model comparison.

```python
class TraceNormalizer:
    def normalize(self, result: TraceResult) -> NormalizedTrace
    def check_consensus(self, normalized: list[NormalizedTrace]) -> bool
```

---

### `Arbiter` (`rde/arbiter.py`)

Resolves disagreements through causal necessity.

```python
class Arbiter:
    def __init__(self, router: ModelRouter, model: str, temperature: float = 0.1) -> None

    async def arbitrate(
        self,
        normalized_traces: list[NormalizedTrace],
        env: ContextEnvironment,
        budget: RecursionBudget | None = None,
    ) -> ArbitrationResult

    def check_convergence(
        self,
        arbitration: ArbitrationResult,
        iteration: int,
        max_iterations: int,
        prior_shadows: list[list[str]] | None = None,
    ) -> ConvergenceResult
```

---

## Data Models (`rde/models.py`)

### Enums

```python
class ConstraintLevel(str, Enum):
    OPEN = "open"
    GUIDED = "guided"
    STRUCTURED = "structured"

class ConfidenceLevel(str, Enum):
    NECESSARY = "necessary"
    CONTINGENT = "contingent"
    UNRESOLVED = "unresolved"
```

### Trace Models

```python
class TraceConfig(BaseModel):
    role: str
    perspective: str
    system_prompt: str
    context_strategy: str = "full"
    temperature: float = 0.7
    model_preference: str = "any"
    recursion_budget: int = 0
    can_recurse: bool = False

class TraceResult(BaseModel):
    trace_id: str
    role: str
    model_used: str
    raw_output: str
    extracted_answer: Optional[str] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    token_usage: dict = Field(default_factory=dict)

class NormalizedTrace(BaseModel):
    trace_id: str
    role: str
    model_family: str
    conclusion: str
    reasoning_chain: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    evidence_cited: list[str] = Field(default_factory=list)
    raw_output: str = ""
```

### Arbitration Models

```python
class ArbitrationResult(BaseModel):
    resolution: str
    causal_chain: str
    confidence: ConfidenceLevel
    interference_detected: list[str] = Field(default_factory=list)
    traces_adopted: list[str] = Field(default_factory=list)
    traces_rejected: list[str] = Field(default_factory=list)
    shadows: list[str] = Field(default_factory=list)
    raw_output: str = ""

class ConvergenceResult(BaseModel):
    should_stop: bool
    reason: str
    next_framings: list[str] = Field(default_factory=list)
    shadow_report_for_human: str = ""
```

### Budget & Call Tree

```python
class RecursionBudget(BaseModel):
    max_depth: int = 3
    max_total_calls: int = 20
    max_cost_usd: float = 10.0
    current_depth: int = 0
    total_calls: int = 0
    total_cost_usd: float = 0.0

    def can_recurse(self) -> bool
    def child_budget(self) -> RecursionBudget
    def record_call(self, cost_usd: float = 0.0) -> None

class CallTreeNode(BaseModel):
    node_type: str  # "trace" | "arbiter" | "sub_lm" | "sub_dialectic"
    model: str = ""
    role: str = ""
    depth: int = 0
    input_summary: str = ""
    output_summary: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    children: list[CallTreeNode] = Field(default_factory=list)
```

### Engine Result

```python
class EngineResult(BaseModel):
    resolution: str
    confidence: ConfidenceLevel
    shadows: list[str] = Field(default_factory=list)
    causal_chain: str = ""
    iterations: int = 1
    trace_results: list[TraceResult] = Field(default_factory=list)
    normalized_traces: list[NormalizedTrace] = Field(default_factory=list)
    arbitration: Optional[ArbitrationResult] = None
    total_latency_ms: float = 0.0
    consensus_reached: bool = False
    call_tree: Optional[CallTreeNode] = None
```

### Configuration

```python
class ModelConfig(BaseModel):
    orchestrator_model: str = "claude-opus-4-6"
    arbiter_model: str = "claude-opus-4-6"
    trace_models: list[str] = [
        "claude-sonnet-4-5-20250929", "gpt-5", "gemini-2.5-pro"
    ]
    sub_lm_models: list[str] = [
        "claude-haiku-4-5-20251001", "gpt-5-mini"
    ]
    trace_assignment: str = "round_robin"
    scaffolding_preference: str = "auto"
    arbiter_temperature: float = 0.1
    orchestrator_temperature: float = 0.3
    sub_lm_temperature: float = 0.3
    local_model_path: str = "~/Models/Qwen3-8B-4bit"
```

---

## Provider System

### `BaseProvider` (`rde/providers/base.py`)

```python
@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    latency_ms: float = 0.0

class BaseProvider(ABC):
    async def complete(
        self, messages, model, temperature=0.7,
        max_tokens=4096, response_format=None,
    ) -> LLMResponse

    def supports_model(self, model: str) -> bool
    async def close(self) -> None
```

### `ModelRouter` (`rde/providers/router.py`)

```python
class ModelRouter:
    def __init__(self, config: ModelConfig | None = None) -> None

    def get_provider(self, model: str) -> BaseProvider
    async def complete(
        self, messages, model, temperature=0.7,
        max_tokens=4096, response_format=None,
    ) -> LLMResponse
    def assign_trace_models(self, traces: list[TraceConfig]) -> list[str]
    async def close(self) -> None

    @property
    def available_providers(self) -> list[str]
```

---

## Independence Metrics (`rde/utils/metrics.py`)

```python
class PairwiseMetric(BaseModel):
    trace_a: str
    trace_b: str
    metric_name: str
    value: float
    detail: str = ""

class IndependenceReport(BaseModel):
    conclusion_agreement: float = 0.0
    model_diversity_score: float = 0.0
    avg_reasoning_divergence: float = 0.0
    avg_jaccard_distance: float = 0.0
    agreement_for_different_reasons_count: int = 0
    pairwise_metrics: list[PairwiseMetric] = []
    embedding_distances: list[PairwiseMetric] = []
    summary: str = ""

# Functions
def conclusion_agreement(traces: list[NormalizedTrace]) -> float
def reasoning_chain_divergence(traces: list[NormalizedTrace]) -> list[PairwiseMetric]
def jaccard_distance(traces: list[NormalizedTrace]) -> list[PairwiseMetric]
def agreement_for_different_reasons(
    traces: list[NormalizedTrace], divergence_threshold: float = 0.3,
) -> list[PairwiseMetric]
def model_diversity_score(traces: list[NormalizedTrace]) -> float
async def embedding_distance(
    traces: list[NormalizedTrace], google_api_key: str | None = None,
) -> list[PairwiseMetric] | None
async def compute_all(
    traces: list[NormalizedTrace], include_embeddings: bool = True,
) -> IndependenceReport
```

---

## Utilities

### Extraction (`rde/utils/extraction.py`)

```python
def extract_boxed(text: str) -> str | None
def extract_json_block(text: str) -> str | None
```

### Visualizer (`rde/utils/visualizer.py`)

```python
def call_tree_to_json(tree: CallTreeNode) -> str
def call_tree_to_ascii(tree: CallTreeNode, indent: int = 0) -> str
def save_call_tree(tree: CallTreeNode, path: str) -> None
```

---

## Sandbox System (`rde/sandbox/`)

```python
@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    runtime_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def success(self) -> bool

class BaseSandbox(ABC):
    async def execute(
        self, code: str, language: str = "python",
        timeout_seconds: float = 30.0, env: dict | None = None,
    ) -> ExecutionResult
    async def close(self) -> None

class LocalSandbox(BaseSandbox):  # Functional
class ModalSandbox(BaseSandbox):  # Stub
class E2BSandbox(BaseSandbox):    # Stub
```

---

## Training Pipeline (`rde/training/`)

### Models (`rde/training/models.py`)

```python
class DistillationStrategy(str, Enum):
    RESOLUTION = "resolution"
    FULL_DIALECTIC = "full_dialectic"
    ROLE_SPECIFIC = "role_specific"

class ProviderFormat(str, Enum):
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    JSONL_GENERIC = "jsonl_generic"

class TrainingExample(BaseModel):
    input_text: str
    output_text: str
    strategy: DistillationStrategy
    source_problem: str = ""
    source_run_index: int = 0
    metadata: dict = Field(default_factory=dict)

class CollectedResult(BaseModel):
    problem_name: str
    problem_prompt: str
    run_index: int = 0
    engine_result: dict = Field(default_factory=dict)
    independence_report: Optional[dict] = None
    collection_timestamp: str = ""

class CollectionConfig(BaseModel):
    problems: list[dict] | None = None
    output_path: str = "training_data/engine_results.jsonl"
    use_orchestrator: bool = True
    max_iterations: int = 1
    max_depth: int = 3
    include_metrics: bool = True
    runs_per_problem: int = 1

class FormattingConfig(BaseModel):
    input_path: str = "training_data/engine_results.jsonl"
    output_path: str = "training_data/formatted"
    strategies: list[DistillationStrategy] = [DistillationStrategy.RESOLUTION]
    provider_format: ProviderFormat = ProviderFormat.GEMINI
    min_trace_count: int = 2
    require_arbitration: bool = False

class EvaluationConfig(BaseModel):
    finetuned_model: str
    baseline_config: dict = Field(default_factory=dict)
    problems: list[dict] = Field(default_factory=list)
    strategies_to_evaluate: list[DistillationStrategy] = [DistillationStrategy.RESOLUTION]

class EvaluationResult(BaseModel):
    strategy: DistillationStrategy
    problem_name: str
    finetuned_confidence: str
    baseline_confidence: str
    finetuned_avg_divergence: float
    baseline_avg_divergence: float
    divergence_delta: float
    finetuned_avg_jaccard: float
    baseline_avg_jaccard: float
    jaccard_delta: float
```

### Collector (`rde/training/collector.py`)

```python
TRAINING_PROBLEMS: list[dict[str, str]]  # 20 problems

async def collect(
    config: CollectionConfig, model_config: ModelConfig | None = None,
) -> list[CollectedResult]

async def collect_single(
    problem: dict, engine: DialecticalEngine, run_index: int = 0,
    max_iterations: int = 1, max_depth: int = 3, include_metrics: bool = True,
) -> CollectedResult

def save_results(results: list[CollectedResult], path: str | Path) -> None
def load_results(path: str | Path) -> list[CollectedResult]
```

### Formatter (`rde/training/formatter.py`)

```python
def format_resolution(result: CollectedResult) -> TrainingExample | None
def format_full_dialectic(result: CollectedResult) -> TrainingExample | None
def format_role_specific(result: CollectedResult) -> list[TrainingExample]

def to_gemini_format(examples: list[TrainingExample]) -> list[dict]
def to_anthropic_format(examples: list[TrainingExample]) -> list[dict]
def to_generic_jsonl(examples: list[TrainingExample]) -> list[dict]

def format_all(
    results: list[CollectedResult], config: FormattingConfig,
) -> list[TrainingExample]

def save_formatted(
    examples: list[TrainingExample], path: str | Path, provider_format: ProviderFormat,
) -> None
```

### Evaluator (`rde/training/evaluator.py`)

```python
async def evaluate(config: EvaluationConfig) -> list[EvaluationResult]
async def run_finetuned(problem: dict, finetuned_model: str, strategy: DistillationStrategy) -> tuple
async def run_baseline(problem: dict, baseline_config: ModelConfig) -> tuple
def compute_deltas(ft_report, bl_report) -> dict[str, float]
def format_evaluation_table(results: list[EvaluationResult]) -> str
def save_evaluation_results(results: list[EvaluationResult], path: str | Path) -> None
```

---

## CLI (`rde/cli.py`)

```
rde "prompt"                        # Basic usage
rde --stdin                         # Read from stdin
rde --no-orchestrator               # Skip orchestrator, use defaults
rde --orchestrator-model MODEL      # Custom orchestrator model
rde --arbiter-model MODEL           # Custom arbiter model
rde --trace-models "a,b,c"          # Custom trace models
rde --max-iterations N              # Multi-iteration reframing
rde --max-depth N                   # Recursion depth limit
rde --trace-log PATH                # Save call tree JSON
rde --metrics                       # Show independence metrics
rde --json                          # JSON output
rde -v / --verbose                  # Debug logging
```
