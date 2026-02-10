"""Benchmark dataset loaders for OOLONG, OOLONG-Pairs, and S-NIAH.

OOLONG: Loads from the official oolongbench/oolong-synth HuggingFace dataset
(Bertsch et al., 2511.02817). Pre-constructed questions with per-answer-type scoring.

OOLONG-Pairs: Constructs TREC-coarse contexts in OOLONG format, applies the
20 pairwise queries from RLM paper Appendix D.1 (Zhang et al., 2512.24601).

S-NIAH: Generates RULER-style needle-in-haystack tasks (Hsieh et al., 2024).
"""

from __future__ import annotations

import ast
import datetime
import logging
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Approximate chars per token (conservative)
CHARS_PER_TOKEN = 4

# TREC-coarse label mapping (short → full name used in OOLONG-Pairs queries)
TREC_LABEL_FULL = {
    "ABBR": "abbreviation",
    "DESC": "description and abstract concept",
    "ENTY": "entity",
    "HUM": "human being",
    "LOC": "location",
    "NUM": "numeric value",
}

# Reverse mapping
TREC_LABEL_SHORT = {v: k for k, v in TREC_LABEL_FULL.items()}


@dataclass
class BenchmarkTask:
    """A single benchmark evaluation task."""

    task_id: str
    context: str  # The long context text
    query: str  # The question to answer
    ground_truth: Any  # Expected answer (varies by benchmark)
    context_tokens: int  # Approximate token count
    scoring_fn: str  # "oolong" | "f1_pairs" | "exact_match"
    metadata: dict = field(default_factory=dict)
    answer_type: str = ""  # For OOLONG: "ANSWER_TYPE.NUMERIC", etc.


@dataclass
class BenchmarkDataset:
    """A collection of benchmark tasks."""

    name: str
    tasks: list[BenchmarkTask]

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)


# ---------------------------------------------------------------------------
# OOLONG — official HuggingFace dataset
# ---------------------------------------------------------------------------

def _parse_oolong_answer(answer_str: str) -> Any:
    """Parse answer from oolong-synth stored format.

    Answers are stored as Python literal strings, e.g.:
      "[7]" → 7, "['incorrect']" → 'incorrect',
      "[datetime.date(2023, 10, 21)]" → datetime.date(2023, 10, 21)
    """
    try:
        if "datetime" in answer_str:
            namespace = {"datetime": datetime}
            result = eval(answer_str, {"__builtins__": {}}, namespace)  # noqa: S307
            return result[0]
        return ast.literal_eval(answer_str)[0]
    except (ValueError, SyntaxError, IndexError):
        return answer_str


class OolongLoader:
    """Load OOLONG benchmark from the official oolongbench/oolong-synth dataset."""

    @staticmethod
    def load(
        context_len: int = 131072,
        split: str = "test",
        max_tasks: int | None = None,
        seed: int = 42,
        dataset_filter: str | None = None,
    ) -> BenchmarkDataset:
        """Load OOLONG tasks at the specified context length.

        Args:
            context_len: Target context length in tokens. Valid values are powers
                of 2 from 1024 to 4194304. The RLM comparison point is 131072.
            split: Dataset split ("test" has 5200 rows, "validation" has 1300).
            max_tasks: Limit number of tasks (for dry runs / testing).
            seed: Random seed for task selection when max_tasks is set.
            dataset_filter: Filter to specific source dataset (e.g., "agnews").
        """
        from datasets import load_dataset

        ds = load_dataset("oolongbench/oolong-synth", split=split)

        # Filter by context length
        ds = ds.filter(lambda x: x["context_len"] == context_len)
        logger.info(
            "Loaded %d OOLONG tasks at context_len=%d from %s split",
            len(ds), context_len, split,
        )

        if dataset_filter:
            ds = ds.filter(lambda x: x["dataset"] == dataset_filter)

        if max_tasks and max_tasks < len(ds):
            rng = random.Random(seed)
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            ds = ds.select(indices[:max_tasks])

        tasks = []
        for item in ds:
            ground_truth = _parse_oolong_answer(item["answer"])
            answer_type = item["answer_type"]

            tasks.append(BenchmarkTask(
                task_id=f"oolong_{item['id']}",
                context=item["context_window_text"],
                query=item["question"],
                ground_truth=ground_truth,
                context_tokens=item["context_len"],
                scoring_fn="oolong",
                answer_type=answer_type,
                metadata={
                    "task_type": item["task"],
                    "task_group": item["task_group"],
                    "dataset": item["dataset"],
                    "context_window_id": item["context_window_id"],
                    "num_labels": item["num_labels"],
                },
            ))

        return BenchmarkDataset(name="oolong", tasks=tasks)

    # Keep backward-compatible alias for tests that used the old API
    @staticmethod
    def load_trec_coarse(
        context_size_tokens: int = 131_000,
        num_tasks: int = 50,
        seed: int = 42,
        cache_dir: Path | None = None,
    ) -> BenchmarkDataset:
        """Backward-compatible wrapper — loads from HF or falls back to synthetic."""
        # Map approximate token counts to exact powers of 2
        context_len_map = {
            500: 1024, 1000: 1024, 1024: 1024, 2000: 2048, 2048: 2048,
            4000: 4096, 4096: 4096, 8000: 8192, 16000: 16384,
            32000: 32768, 64000: 65536, 131000: 131072, 131072: 131072,
        }
        context_len = context_len_map.get(context_size_tokens, 1024)

        try:
            return OolongLoader.load(
                context_len=context_len,
                max_tasks=num_tasks,
                seed=seed,
            )
        except Exception as exc:
            logger.warning("HF OOLONG load failed (%s), using synthetic", exc)
            return _build_synthetic_oolong(num_tasks, context_size_tokens, seed)

    # Also keep backward-compatible alias for pairs
    @staticmethod
    def load_oolong_pairs(
        context_size_tokens: int = 32_000,
        num_tasks: int = 20,
        seed: int = 42,
        cache_dir: Path | None = None,
    ) -> BenchmarkDataset:
        """Backward-compatible wrapper for OOLONG-Pairs."""
        return OolongPairsLoader.load(
            context_size_tokens=context_size_tokens,
            num_tasks=num_tasks,
            seed=seed,
        )


def _build_synthetic_oolong(
    num_tasks: int, context_size_tokens: int, seed: int,
) -> BenchmarkDataset:
    """Fallback: synthetic OOLONG-like tasks when HF is unavailable."""
    rng = random.Random(seed)
    tasks = []
    for i in range(num_tasks):
        # Generate a small synthetic context with a counting question
        n_items = max(10, context_size_tokens // 50)
        labels = list(TREC_LABEL_FULL.values())
        items = [rng.choice(labels) for _ in range(n_items)]
        context = "\n".join(f"Item {j}: category={item}" for j, item in enumerate(items))
        count = sum(1 for x in items if x == items[0])
        tasks.append(BenchmarkTask(
            task_id=f"oolong_synth_{i:03d}",
            context=context,
            query=f"How many items have the label '{items[0]}'? Give your final answer in the form 'Answer: number'.",
            ground_truth=count,
            context_tokens=context_size_tokens,
            scoring_fn="oolong",
            answer_type="ANSWER_TYPE.NUMERIC",
        ))
    return BenchmarkDataset(name="oolong", tasks=tasks)


# ---------------------------------------------------------------------------
# OOLONG-Pairs — TREC-coarse + 20 RLM paper queries (Appendix D.1)
# ---------------------------------------------------------------------------

# Common suffix for all 20 OOLONG-Pairs queries
_PAIRS_LABEL_INSTRUCTION = (
    "Each of the questions can be labelled as one of the labels "
    "(the data does not provide the labels, you need to figure out "
    "the label from the semantics of the question): "
    "description and abstract concept, entity, human being, "
    "numeric value, location, abbreviation. "
    "In your answer, list all pairs in the format "
    "(user_id_1, user_id_2), separated by newlines."
)

_PAIRS_PREFIX = (
    "In the above data, list all pairs of user IDs "
    "(no duplicate pairs, list lower ID first) "
)


def _symmetric_condition(labels: set[str]):
    """Both users have ≥1 instance with any of the given labels."""
    def check(entries):
        return any(e["label"] in labels for e in entries)
    return check


def _temporal_condition(label: str, direction: str, cutoff: datetime.date):
    """All instances of 'label' must be before/after cutoff date."""
    def check(entries):
        label_entries = [e for e in entries if e["label"] == label]
        if not label_entries:
            return True  # No entries to violate constraint
        if direction == "after":
            return all(e["date"] > cutoff for e in label_entries)
        return all(e["date"] < cutoff for e in label_entries)
    return check


def _at_least(entries, label: str, n: int) -> bool:
    return sum(1 for e in entries if e["label"] == label) >= n


def _exactly(entries, label: str, n: int) -> bool:
    return sum(1 for e in entries if e["label"] == label) == n


# Define the 20 queries from RLM Appendix D.1
# Each query has: id, constraint_text, type, and constraint functions
OOLONG_PAIRS_QUERIES = [
    # Tasks 1-3, 6, 8: Simple symmetric — both users have ≥1 of either label
    {
        "id": 1,
        "constraint": "where both users have at least one instance with a numeric value or location",
        "type": "symmetric",
        "check_user": _symmetric_condition({"numeric value", "location"}),
    },
    {
        "id": 2,
        "constraint": "where both users have at least one instance with an entity or human being",
        "type": "symmetric",
        "check_user": _symmetric_condition({"entity", "human being"}),
    },
    {
        "id": 3,
        "constraint": "where both users have at least one instance with a description and abstract concept or abbreviation",
        "type": "symmetric",
        "check_user": _symmetric_condition({"description and abstract concept", "abbreviation"}),
    },
    # Tasks 4-5, 7, 9-10: Symmetric + temporal constraint
    {
        "id": 4,
        "constraint": (
            "where both users have at least one instance with a human being or location, "
            "and all instances that are a human being for both users must be after January 6, 2023"
        ),
        "type": "symmetric_temporal",
        "check_user": _symmetric_condition({"human being", "location"}),
        "check_temporal": _temporal_condition("human being", "after", datetime.date(2023, 1, 6)),
    },
    {
        "id": 5,
        "constraint": (
            "where both users have at least one instance with an entity or numeric value, "
            "and all instances that are an entity for both users must be before March 15, 2023"
        ),
        "type": "symmetric_temporal",
        "check_user": _symmetric_condition({"entity", "numeric value"}),
        "check_temporal": _temporal_condition("entity", "before", datetime.date(2023, 3, 15)),
    },
    {
        "id": 6,
        "constraint": "where both users have at least one instance with a location or abbreviation",
        "type": "symmetric",
        "check_user": _symmetric_condition({"location", "abbreviation"}),
    },
    {
        "id": 7,
        "constraint": (
            "where both users have at least one instance with a description and abstract concept "
            "or numeric value, and all instances that are a numeric value for both users must be "
            "after February 1, 2023"
        ),
        "type": "symmetric_temporal",
        "check_user": _symmetric_condition({"description and abstract concept", "numeric value"}),
        "check_temporal": _temporal_condition("numeric value", "after", datetime.date(2023, 2, 1)),
    },
    {
        "id": 8,
        "constraint": "where both users have at least one instance with a human being or description and abstract concept",
        "type": "symmetric",
        "check_user": _symmetric_condition({"human being", "description and abstract concept"}),
    },
    {
        "id": 9,
        "constraint": (
            "where both users have at least one instance with an entity or location, "
            "and all instances that are a location for both users must be after April 10, 2023"
        ),
        "type": "symmetric_temporal",
        "check_user": _symmetric_condition({"entity", "location"}),
        "check_temporal": _temporal_condition("location", "after", datetime.date(2023, 4, 10)),
    },
    {
        "id": 10,
        "constraint": (
            "where both users have at least one instance with a numeric value or abbreviation, "
            "and all instances that are an abbreviation for both users must be before May 20, 2023"
        ),
        "type": "symmetric_temporal",
        "check_user": _symmetric_condition({"numeric value", "abbreviation"}),
        "check_temporal": _temporal_condition("abbreviation", "before", datetime.date(2023, 5, 20)),
    },
    # Tasks 11-20: Asymmetric — different conditions per user
    {
        "id": 11,
        "constraint": (
            "such that one user has at least one instance with entity and one with abbreviation, "
            "and the other user has exactly one instance with entity"
        ),
        "type": "asymmetric",
        "cond_a": lambda e: _at_least(e, "entity", 1) and _at_least(e, "abbreviation", 1),
        "cond_b": lambda e: _exactly(e, "entity", 1),
    },
    {
        "id": 12,
        "constraint": (
            "such that one user has at least two instances with numeric value, "
            "and the other user has at least one instance with location and at least one instance "
            "with human being"
        ),
        "type": "asymmetric",
        "cond_a": lambda e: _at_least(e, "numeric value", 2),
        "cond_b": lambda e: _at_least(e, "location", 1) and _at_least(e, "human being", 1),
    },
    {
        "id": 13,
        "constraint": (
            "such that one user has exactly one instance with description and abstract concept, "
            "and the other user has at least one instance with abbreviation and at least one "
            "instance with entity"
        ),
        "type": "asymmetric",
        "cond_a": lambda e: _exactly(e, "description and abstract concept", 1),
        "cond_b": lambda e: _at_least(e, "abbreviation", 1) and _at_least(e, "entity", 1),
    },
    {
        "id": 14,
        "constraint": (
            "such that one user has at least one instance with human being and at least one "
            "instance with numeric value, and the other user has exactly two instances with location"
        ),
        "type": "asymmetric",
        "cond_a": lambda e: _at_least(e, "human being", 1) and _at_least(e, "numeric value", 1),
        "cond_b": lambda e: _exactly(e, "location", 2),
    },
    {
        "id": 15,
        "constraint": (
            "such that one user has at least one instance with entity, at least one instance "
            "with location, and at least one instance with abbreviation, and the other user has "
            "exactly one instance with numeric value"
        ),
        "type": "asymmetric",
        "cond_a": lambda e: (
            _at_least(e, "entity", 1) and _at_least(e, "location", 1) and _at_least(e, "abbreviation", 1)
        ),
        "cond_b": lambda e: _exactly(e, "numeric value", 1),
    },
    {
        "id": 16,
        "constraint": (
            "such that one user has at least one instance with description and abstract concept "
            "and at least one instance with human being, and the other user has at least two "
            "instances with entity and exactly one instance with abbreviation"
        ),
        "type": "asymmetric",
        "cond_a": lambda e: (
            _at_least(e, "description and abstract concept", 1) and _at_least(e, "human being", 1)
        ),
        "cond_b": lambda e: _at_least(e, "entity", 2) and _exactly(e, "abbreviation", 1),
    },
    {
        "id": 17,
        "constraint": (
            "such that one user has exactly one instance with numeric value, "
            "and the other user has at least one instance with location and at least one "
            "instance with description and abstract concept"
        ),
        "type": "asymmetric",
        "cond_a": lambda e: _exactly(e, "numeric value", 1),
        "cond_b": lambda e: _at_least(e, "location", 1) and _at_least(e, "description and abstract concept", 1),
    },
    {
        "id": 18,
        "constraint": (
            "such that one user has at least one instance with abbreviation and exactly one "
            "instance with human being, and the other user has at least one instance with entity "
            "and at least one instance with numeric value"
        ),
        "type": "asymmetric",
        "cond_a": lambda e: _at_least(e, "abbreviation", 1) and _exactly(e, "human being", 1),
        "cond_b": lambda e: _at_least(e, "entity", 1) and _at_least(e, "numeric value", 1),
    },
    {
        "id": 19,
        "constraint": (
            "such that one user has at least two instances with location and at least one "
            "instance with entity, and the other user has exactly one instance with description "
            "and abstract concept and exactly one instance with abbreviation"
        ),
        "type": "asymmetric",
        "cond_a": lambda e: _at_least(e, "location", 2) and _at_least(e, "entity", 1),
        "cond_b": lambda e: (
            _exactly(e, "description and abstract concept", 1) and _exactly(e, "abbreviation", 1)
        ),
    },
    {
        "id": 20,
        "constraint": (
            "such that one user has at least one instance with numeric value and at least one "
            "instance with human being, and the other user has at least one instance with "
            "location, at least one instance with entity, and exactly one instance with abbreviation"
        ),
        "type": "asymmetric",
        "cond_a": lambda e: _at_least(e, "numeric value", 1) and _at_least(e, "human being", 1),
        "cond_b": lambda e: (
            _at_least(e, "location", 1) and _at_least(e, "entity", 1)
            and _exactly(e, "abbreviation", 1)
        ),
    },
]


def _build_query_text(query_spec: dict) -> str:
    """Build the full query text from a query spec."""
    return f"{_PAIRS_PREFIX}{query_spec['constraint']}. {_PAIRS_LABEL_INSTRUCTION}"


def _check_pair(query_spec: dict, user_a_entries: list, user_b_entries: list) -> bool:
    """Check whether a pair of users satisfies the query constraint."""
    qtype = query_spec["type"]

    if qtype == "symmetric":
        return query_spec["check_user"](user_a_entries) and query_spec["check_user"](user_b_entries)

    if qtype == "symmetric_temporal":
        base_ok = (
            query_spec["check_user"](user_a_entries)
            and query_spec["check_user"](user_b_entries)
        )
        if not base_ok:
            return False
        return (
            query_spec["check_temporal"](user_a_entries)
            and query_spec["check_temporal"](user_b_entries)
        )

    if qtype == "asymmetric":
        cond_a, cond_b = query_spec["cond_a"], query_spec["cond_b"]
        return (
            (cond_a(user_a_entries) and cond_b(user_b_entries))
            or (cond_b(user_a_entries) and cond_a(user_b_entries))
        )

    return False


class OolongPairsLoader:
    """Construct OOLONG-Pairs tasks from TREC-coarse data + RLM paper queries."""

    @staticmethod
    def load(
        context_size_tokens: int = 32_768,
        num_tasks: int = 20,
        seed: int = 42,
    ) -> BenchmarkDataset:
        """Build OOLONG-Pairs benchmark tasks.

        Constructs a TREC-coarse context in OOLONG format (Date || User || Instance),
        then applies the 20 pairwise queries from Zhang et al. Appendix D.1.

        Args:
            context_size_tokens: Target context size in tokens.
            num_tasks: Number of tasks (max 20, one per query).
            seed: Random seed for reproducible context construction.
        """
        rng = random.Random(seed)
        documents = _load_trec_documents(rng)

        target_chars = context_size_tokens * CHARS_PER_TOKEN
        entries = _build_oolong_context_entries(documents, target_chars, rng)

        # Build the context string (without labels)
        n = len(entries)
        preamble = (
            f"The following lines contain {n} general-knowledge questions, "
            "one per line. Each line has a User ID, which is not necessarily "
            "unique, i.e. each User ID can be associated with multiple questions.\n\n"
            "You will be asked to answer questions about the aggregate statistics "
            f"across all {n} questions in this dataset. Do not try to guess, "
            "estimate, or approximate the result. Calculate the exact answer "
            "given these datapoints.\n\n"
        )
        body = "\n".join(
            f"Date: {e['date'].strftime('%b %d, %Y')} || User: {e['user_id']} || Instance: {e['text']}"
            for e in entries
        )
        epilogue = (
            f"\n\nRecall: the preceding lines contain {n} general-knowledge questions, "
            "one per line, each with a User ID and Date."
        )
        context = preamble + body + epilogue

        # Group entries by user for ground truth computation
        user_entries: dict[str, list] = {}
        for e in entries:
            uid = str(e["user_id"])
            user_entries.setdefault(uid, []).append(e)

        user_ids = sorted(user_entries.keys())

        # Build tasks from the 20 queries
        tasks = []
        for query_spec in OOLONG_PAIRS_QUERIES[:num_tasks]:
            query_text = _build_query_text(query_spec)

            # Compute ground truth pairs
            gt_pairs: set[tuple[str, str]] = set()
            for i, uid_a in enumerate(user_ids):
                for uid_b in user_ids[i + 1:]:
                    if _check_pair(query_spec, user_entries[uid_a], user_entries[uid_b]):
                        gt_pairs.add((uid_a, uid_b))

            tasks.append(BenchmarkTask(
                task_id=f"oolong_pairs_{query_spec['id']:03d}",
                context=context,
                query=query_text,
                ground_truth=gt_pairs,
                context_tokens=len(context) // CHARS_PER_TOKEN,
                scoring_fn="f1_pairs",
                metadata={
                    "num_entries": n,
                    "num_users": len(user_ids),
                    "num_gt_pairs": len(gt_pairs),
                    "query_type": query_spec["type"],
                },
            ))

        return BenchmarkDataset(name="oolong_pairs", tasks=tasks)


def _load_trec_documents(rng: random.Random) -> list[dict]:
    """Load TREC-coarse documents from HuggingFace or generate synthetic."""
    try:
        from datasets import load_dataset
        ds = load_dataset("lukasgarbas/trec", split="train")
        documents = []
        for item in ds:
            label = item["coarse_label"]
            if label in TREC_LABEL_FULL:
                documents.append({
                    "text": item["text"],
                    "label": TREC_LABEL_FULL[label],  # Full name for query matching
                })
        logger.info("Loaded %d TREC documents from HuggingFace", len(documents))
        return documents
    except Exception as exc:
        logger.warning("TREC load failed (%s), generating synthetic data", exc)
        return _generate_synthetic_trec(rng)


def _generate_synthetic_trec(rng: random.Random, num_docs: int = 5000) -> list[dict]:
    """Generate synthetic TREC-like question classification documents."""
    templates = {
        "abbreviation": [
            "What does {w} stand for?",
            "What is the abbreviation for {w}?",
        ],
        "description and abstract concept": [
            "How does {w} work?",
            "What causes {w}?",
            "What is the definition of {w}?",
        ],
        "entity": [
            "What {w} is the largest?",
            "Name a famous {w}.",
        ],
        "human being": [
            "Who invented {w}?",
            "Who was the first person to {w}?",
        ],
        "location": [
            "Where is {w} located?",
            "What country produces {w}?",
        ],
        "numeric value": [
            "How many {w} are there?",
            "What year did {w} happen?",
        ],
    }
    fillers = [
        "photosynthesis", "democracy", "algorithm", "telescope", "continent",
        "molecule", "symphony", "glacier", "volcano", "equation",
    ]
    labels = list(templates.keys())
    docs = []
    for _ in range(num_docs):
        label = rng.choice(labels)
        text = rng.choice(templates[label]).format(w=rng.choice(fillers))
        docs.append({"text": text, "label": label})
    return docs


def _build_oolong_context_entries(
    documents: list[dict],
    target_chars: int,
    rng: random.Random,
) -> list[dict]:
    """Build OOLONG-format entries with Date, User ID, and Instance text.

    User IDs follow a Zipf distribution (80% of entries from ~20% of users).
    Dates span Oct 2022 to May 2025.
    """
    rng.shuffle(documents)

    # Determine how many entries fit
    packed = []
    total_chars = 0
    for doc in documents:
        entry_chars = len(doc["text"]) + 60  # overhead for Date/User/Instance formatting
        if total_chars + entry_chars > target_chars:
            break
        packed.append(doc)
        total_chars += entry_chars

    n = len(packed)
    if n == 0:
        return []

    # Generate Zipf-distributed user IDs
    num_users = max(5, n // 20)  # ~5% unique users
    user_pool = [rng.randint(10000, 99999) for _ in range(num_users)]
    weights = [1.0 / (i + 1) for i in range(num_users)]
    assigned_users = rng.choices(user_pool, weights=weights, k=n)

    # Generate random dates spanning Oct 2022 — May 2025
    start_date = datetime.date(2022, 10, 1)
    end_date = datetime.date(2025, 5, 31)
    delta_days = (end_date - start_date).days
    assigned_dates = [
        start_date + datetime.timedelta(days=rng.randint(0, delta_days))
        for _ in range(n)
    ]

    entries = []
    for i, doc in enumerate(packed):
        entries.append({
            "text": doc["text"],
            "label": doc["label"],
            "user_id": assigned_users[i],
            "date": assigned_dates[i],
        })
    return entries


# ---------------------------------------------------------------------------
# S-NIAH — RULER-style needle-in-haystack (Hsieh et al., 2024)
# ---------------------------------------------------------------------------

# Adjective-noun pairs for key generation (RULER uses wonderwords library)
_ADJECTIVES = [
    "cheerful", "bright", "gentle", "swift", "calm", "bold", "quiet", "warm",
    "fierce", "clever", "brave", "eager", "proud", "lucky", "witty", "sleek",
    "rustic", "vivid", "crisp", "sturdy", "lively", "serene", "noble", "merry",
]
_NOUNS = [
    "mountain", "river", "forest", "ocean", "valley", "meadow", "canyon", "island",
    "glacier", "prairie", "desert", "lagoon", "summit", "plateau", "harbor", "ridge",
    "volcano", "tundra", "savanna", "fjord", "delta", "basin", "cavern", "bluff",
]

# Essay-style filler paragraphs (replacing Paul Graham essays)
_ESSAY_TOPICS = [
    "The history of maritime navigation spans thousands of years, from early Polynesian wayfinders "
    "who read ocean swells to modern satellite positioning systems. Each era brought innovations "
    "that expanded the horizons of human exploration and trade.",
    "Agricultural practices vary widely across different climates and geographies. In tropical "
    "regions, slash-and-burn cultivation gave way to sustainable agroforestry systems, while "
    "temperate zones developed sophisticated crop rotation techniques.",
    "The development of printing technology transformed information sharing across civilizations. "
    "From woodblock printing in Tang Dynasty China to Gutenberg's movable type, each advance "
    "democratized access to knowledge.",
    "Mountain ecosystems support remarkably diverse biological communities despite harsh conditions. "
    "Alpine meadows, cloud forests, and subalpine zones each harbor species found nowhere else, "
    "adapted to extreme temperature swings and thin atmosphere.",
    "Urban planning in the 21st century faces unprecedented challenges from climate change, "
    "population growth, and inequality. Cities must balance density with livability, economic "
    "vitality with environmental sustainability.",
    "The study of deep-sea organisms reveals surprising adaptations to extreme pressure, darkness, "
    "and cold. Bioluminescent species illuminate the abyss, while chemosynthetic communities "
    "thrive near hydrothermal vents without sunlight.",
    "Archaeological discoveries continue to reshape our understanding of ancient civilizations. "
    "Remote sensing technologies now reveal buried cities and trade routes invisible to surface "
    "surveys, transforming fieldwork methodology.",
    "Renewable energy technologies are advancing at an accelerating pace, driven by both policy "
    "incentives and plummeting costs. Solar photovoltaic efficiency has doubled in a decade, "
    "while offshore wind capacity expands rapidly.",
    "The cognitive development of children follows predictable stages, yet individual variation "
    "is enormous. Language acquisition, theory of mind, and executive function each emerge "
    "on distinct timelines shaped by both biology and environment.",
    "International trade agreements shape economic relationships globally, creating complex "
    "interdependencies between nations. Supply chain resilience has become a key concern "
    "as geopolitical tensions disrupt established patterns.",
    "Quantum computing promises to solve problems intractable for classical machines. From "
    "molecular simulation for drug discovery to optimization of logistics networks, potential "
    "applications span nearly every scientific and industrial domain.",
    "The evolution of written language reflects the cognitive and social complexity of human "
    "societies. Cuneiform, hieroglyphics, and early alphabets each encoded different relationships "
    "between sound, meaning, and visual form.",
]

# Noise text (RULER niah_single_1 uses repeated simple sentences)
_NOISE_TEXT = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. "
    "There and back again. "
)


class SNIAHLoader:
    """Generate RULER-style Single Needle-in-a-Haystack tasks."""

    @staticmethod
    def load(
        context_sizes: list[int] | None = None,
        tasks_per_size: int = 10,
        seed: int = 42,
    ) -> BenchmarkDataset:
        """Generate S-NIAH tasks following RULER benchmark methodology.

        Creates three types of tasks (matching RULER's niah_single_1/2/3):
        - noise + 7-digit numbers
        - essay + 7-digit numbers
        - essay + UUIDs

        Args:
            context_sizes: List of target context sizes in tokens.
                Default: [32000, 131000] matching common evaluation points.
            tasks_per_size: Number of tasks per context size.
            seed: Random seed for reproducibility.
        """
        if context_sizes is None:
            context_sizes = [32_000, 131_000]

        rng = random.Random(seed)
        tasks = []
        task_idx = 0

        # Cycle through the 3 sub-task types
        subtask_configs = [
            {"name": "noise_numbers", "haystack": "noise", "value_type": "numbers"},
            {"name": "essay_numbers", "haystack": "essay", "value_type": "numbers"},
            {"name": "essay_uuids", "haystack": "essay", "value_type": "uuids"},
        ]

        for ctx_size in context_sizes:
            target_chars = ctx_size * CHARS_PER_TOKEN

            for i in range(tasks_per_size):
                config = subtask_configs[i % len(subtask_configs)]

                # Generate key (adjective-noun pair)
                key = f"{rng.choice(_ADJECTIVES)}-{rng.choice(_NOUNS)}"

                # Generate value
                if config["value_type"] == "numbers":
                    value = str(rng.randint(1_000_000, 9_999_999))
                else:
                    value = str(uuid.UUID(int=rng.getrandbits(128)))

                # Build needle
                needle = f'One of the special magic {config["value_type"]} for "{key}" is: {value}.'

                # Build haystack
                if config["haystack"] == "noise":
                    haystack = SNIAHLoader._build_noise_haystack(target_chars)
                else:
                    haystack = SNIAHLoader._build_essay_haystack(rng, target_chars)

                # Insert needle at random depth
                # Use 40 uniformly-spaced depth positions (matching RULER)
                depth = rng.choice([d / 39 for d in range(40)])
                insert_pos = int(len(haystack) * depth)

                # Find a sentence boundary near the insertion point
                boundary = haystack.find(". ", insert_pos)
                if boundary == -1 or boundary > insert_pos + 200:
                    boundary = insert_pos
                else:
                    boundary += 2  # After the period and space

                context_with_needle = (
                    haystack[:boundary] + "\n" + needle + "\n" + haystack[boundary:]
                )

                # For the engine, we split context and query
                # Context = preamble + haystack + needle, Query = the question
                context_part = (
                    f'A special magic {config["value_type"]} is hidden within the following '
                    f'text. Make sure to memorize it. I will quiz you about the '
                    f'{config["value_type"]} afterwards.\n\n'
                    f'{context_with_needle}'
                )
                query_part = (
                    f'What is the special magic {config["value_type"]} for "{key}" '
                    f'mentioned in the provided text?'
                )

                tasks.append(BenchmarkTask(
                    task_id=f"sniah_{ctx_size // 1000}k_{task_idx:03d}",
                    context=context_part,
                    query=query_part,
                    ground_truth=value,
                    context_tokens=len(context_part) // CHARS_PER_TOKEN,
                    scoring_fn="exact_match",
                    metadata={
                        "target_context_size": ctx_size,
                        "needle_depth": depth,
                        "subtask": config["name"],
                        "value_type": config["value_type"],
                        "key": key,
                    },
                ))
                task_idx += 1

        return BenchmarkDataset(name="sniah", tasks=tasks)

    @staticmethod
    def _build_noise_haystack(target_chars: int) -> str:
        """Build a noise haystack (repeated simple sentences)."""
        repeats = (target_chars // len(_NOISE_TEXT)) + 1
        return (_NOISE_TEXT * repeats)[:target_chars]

    @staticmethod
    def _build_essay_haystack(rng: random.Random, target_chars: int) -> str:
        """Build an essay-style haystack from diverse topic paragraphs."""
        paragraphs = []
        total = 0
        while total < target_chars:
            topic = rng.choice(_ESSAY_TOPICS)
            # Expand each topic into a multi-sentence paragraph
            sentences = [topic]
            connectors = [
                "Furthermore", "Additionally", "Moreover", "In contrast",
                "Similarly", "Consequently", "Meanwhile", "Nevertheless",
                "Specifically", "Notably", "Indeed", "However",
            ]
            for _ in range(rng.randint(4, 8)):
                connector = rng.choice(connectors)
                # Generate substantive filler (not gibberish)
                fillers = [
                    "researchers have found significant evidence supporting this view",
                    "historical records indicate patterns consistent with these observations",
                    "recent developments suggest the trend will continue in coming decades",
                    "economic analysis reveals complex interactions between multiple factors",
                    "environmental assessments highlight the urgency of coordinated action",
                    "technological innovations continue to reshape traditional approaches",
                    "comparative studies across regions demonstrate notable variations",
                    "longitudinal data confirms the persistence of these underlying dynamics",
                    "interdisciplinary perspectives offer fresh insights into longstanding questions",
                    "practical applications have already begun to emerge from this research",
                ]
                sentences.append(f"{connector}, {rng.choice(fillers)}.")

            paragraph = " ".join(sentences)
            paragraphs.append(paragraph)
            total += len(paragraph) + 2  # +2 for paragraph separator

        return "\n\n".join(paragraphs)[:target_chars]
