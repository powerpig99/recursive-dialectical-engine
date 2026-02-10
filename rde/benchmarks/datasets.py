"""Benchmark dataset loaders for OOLONG, OOLONG-Pairs, and S-NIAH.

OOLONG and OOLONG-Pairs are constructed from the TREC-coarse question
classification dataset (6 categories). S-NIAH is synthetic.

These benchmarks match those used in the RLM paper (Zhang et al., 2512.24601)
for direct comparison with their Table 1 results.
"""

from __future__ import annotations

import logging
import random
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# TREC-coarse 6 categories
TREC_COARSE_LABELS = ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]

# Approximate chars per token (conservative estimate)
CHARS_PER_TOKEN = 4


@dataclass
class BenchmarkTask:
    """A single benchmark evaluation task."""

    task_id: str
    context: str  # The long context text
    query: str  # The question to answer
    ground_truth: Any  # Expected answer (dict for OOLONG, set for Pairs, str for S-NIAH)
    context_tokens: int  # Approximate token count
    scoring_fn: str  # "oolong_accuracy" | "f1_pairs" | "exact_match"
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkDataset:
    """A collection of benchmark tasks."""

    name: str
    tasks: list[BenchmarkTask]

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)


class OolongLoader:
    """Load or construct OOLONG benchmark tasks from TREC-coarse."""

    @staticmethod
    def load_trec_coarse(
        context_size_tokens: int = 131_000,
        num_tasks: int = 50,
        seed: int = 42,
        cache_dir: Path | None = None,
    ) -> BenchmarkDataset:
        """Construct OOLONG tasks from TREC-coarse documents.

        If the `datasets` library is available, loads from HuggingFace.
        Otherwise, generates synthetic TREC-like documents.

        Each task packs documents into a context of target token count,
        then asks: "What fraction of documents belong to each category?"
        """
        rng = random.Random(seed)
        documents = OolongLoader._load_or_generate_trec_docs(rng, cache_dir)
        target_chars = context_size_tokens * CHARS_PER_TOKEN

        tasks = []
        for i in range(num_tasks):
            # Shuffle and pack documents to target size
            rng.shuffle(documents)
            packed_docs, label_counts = OolongLoader._pack_documents(
                documents, target_chars, rng
            )

            context = "\n\n".join(
                f"[Document {j + 1}]\nCategory: [HIDDEN]\n{doc['text']}"
                for j, doc in enumerate(packed_docs)
            )

            # Ground truth: distribution of categories
            total = sum(label_counts.values())
            ground_truth = {
                label: label_counts.get(label, 0) for label in TREC_COARSE_LABELS
            }

            query = (
                f"This context contains {len(packed_docs)} documents. "
                "Each document belongs to one of these categories: "
                f"{', '.join(TREC_COARSE_LABELS)}. "
                "Classify each document and report the count for each category. "
                "Output a JSON object mapping category names to counts."
            )

            tasks.append(BenchmarkTask(
                task_id=f"oolong_{i:03d}",
                context=context,
                query=query,
                ground_truth=ground_truth,
                context_tokens=len(context) // CHARS_PER_TOKEN,
                scoring_fn="oolong_accuracy",
                metadata={"total_docs": len(packed_docs), "total_count": total},
            ))

        return BenchmarkDataset(name="oolong", tasks=tasks)

    @staticmethod
    def load_oolong_pairs(
        context_size_tokens: int = 131_000,
        num_tasks: int = 20,
        seed: int = 42,
        cache_dir: Path | None = None,
    ) -> BenchmarkDataset:
        """Construct OOLONG-Pairs tasks (pairwise matching by category).

        Each task asks: "Which pairs of documents share the same category?"
        """
        rng = random.Random(seed)
        documents = OolongLoader._load_or_generate_trec_docs(rng, cache_dir)
        target_chars = context_size_tokens * CHARS_PER_TOKEN

        tasks = []
        for i in range(num_tasks):
            rng.shuffle(documents)
            packed_docs, _ = OolongLoader._pack_documents(documents, target_chars, rng)

            context = "\n\n".join(
                f"[Document {j + 1}]\nCategory: [HIDDEN]\n{doc['text']}"
                for j, doc in enumerate(packed_docs)
            )

            # Ground truth: all pairs of documents with same category
            ground_truth_pairs: set[tuple[str, str]] = set()
            for a_idx in range(len(packed_docs)):
                for b_idx in range(a_idx + 1, len(packed_docs)):
                    if packed_docs[a_idx]["label"] == packed_docs[b_idx]["label"]:
                        ground_truth_pairs.add((
                            f"doc_{a_idx + 1}",
                            f"doc_{b_idx + 1}",
                        ))

            query = (
                f"This context contains {len(packed_docs)} documents. "
                "Each document belongs to one of these categories: "
                f"{', '.join(TREC_COARSE_LABELS)}. "
                "Find all pairs of documents that share the same category. "
                'Output a JSON list of pairs, e.g., [["doc_1", "doc_3"], ["doc_2", "doc_5"]].'
            )

            tasks.append(BenchmarkTask(
                task_id=f"oolong_pairs_{i:03d}",
                context=context,
                query=query,
                ground_truth=ground_truth_pairs,
                context_tokens=len(context) // CHARS_PER_TOKEN,
                scoring_fn="f1_pairs",
                metadata={"total_docs": len(packed_docs), "num_pairs": len(ground_truth_pairs)},
            ))

        return BenchmarkDataset(name="oolong_pairs", tasks=tasks)

    @staticmethod
    def _load_or_generate_trec_docs(
        rng: random.Random,
        cache_dir: Path | None = None,
    ) -> list[dict[str, str]]:
        """Try to load TREC from HuggingFace; fall back to synthetic generation."""
        try:
            from datasets import load_dataset
            ds = load_dataset("CogComp/trec", split="train")
            documents = []
            for item in ds:
                label_idx = item["coarse_label"]
                label = TREC_COARSE_LABELS[label_idx]
                documents.append({"text": item["text"], "label": label})
            logger.info("Loaded %d TREC documents from HuggingFace", len(documents))
            return documents
        except Exception:
            logger.warning("Could not load TREC from HuggingFace; generating synthetic data")
            return OolongLoader._generate_synthetic_trec(rng, num_docs=5000)

    @staticmethod
    def _generate_synthetic_trec(
        rng: random.Random,
        num_docs: int = 5000,
    ) -> list[dict[str, str]]:
        """Generate synthetic TREC-like question classification documents."""
        templates = {
            "ABBR": [
                "What does {word} stand for?",
                "What is the abbreviation for {word}?",
                "Define the acronym {word}.",
            ],
            "DESC": [
                "How does {word} work?",
                "What causes {word}?",
                "Explain the process of {word}.",
                "What is the definition of {word}?",
            ],
            "ENTY": [
                "What {word} is the largest?",
                "Name a famous {word}.",
                "What type of {word} is used?",
            ],
            "HUM": [
                "Who invented {word}?",
                "Who was the first person to {word}?",
                "Name the {word} leader.",
            ],
            "LOC": [
                "Where is {word} located?",
                "What country produces {word}?",
                "In which city is {word}?",
            ],
            "NUM": [
                "How many {word} are there?",
                "What year did {word} happen?",
                "How much does {word} cost?",
            ],
        }
        filler_words = [
            "photosynthesis", "democracy", "algorithm", "telescope", "continent",
            "molecule", "symphony", "glacier", "volcano", "equation",
            "manuscript", "satellite", "chromosome", "thermometer", "parliament",
            "archipelago", "catalyst", "hemisphere", "metamorphosis", "Renaissance",
        ]

        documents = []
        for _ in range(num_docs):
            label = rng.choice(TREC_COARSE_LABELS)
            template = rng.choice(templates[label])
            word = rng.choice(filler_words)
            text = template.format(word=word)
            # Add some padding to make documents more realistic in size
            padding = " ".join(rng.choices(filler_words, k=rng.randint(5, 20)))
            documents.append({"text": f"{text} Context: {padding}", "label": label})

        return documents

    @staticmethod
    def _pack_documents(
        documents: list[dict[str, str]],
        target_chars: int,
        rng: random.Random,
    ) -> tuple[list[dict[str, str]], dict[str, int]]:
        """Pack documents into a context of approximately target_chars."""
        packed = []
        total_chars = 0
        label_counts: dict[str, int] = {}

        for doc in documents:
            doc_chars = len(doc["text"]) + 50  # overhead for formatting
            if total_chars + doc_chars > target_chars:
                break
            packed.append(doc)
            total_chars += doc_chars
            label_counts[doc["label"]] = label_counts.get(doc["label"], 0) + 1

        return packed, label_counts


class SNIAHLoader:
    """Construct Single Needle-in-a-Haystack tasks."""

    @staticmethod
    def load(
        context_sizes: list[int] | None = None,
        tasks_per_size: int = 10,
        seed: int = 42,
    ) -> BenchmarkDataset:
        """Generate S-NIAH tasks with needles hidden in random text.

        Each task has a "needle" (a specific fact) inserted at a random
        position in a "haystack" of filler text.
        """
        if context_sizes is None:
            context_sizes = [32_000, 131_000]

        rng = random.Random(seed)
        tasks = []

        needles = [
            ("The secret project code is PHOENIX-7749.", "PHOENIX-7749"),
            ("The meeting is scheduled for March 15th at 3pm.", "March 15th at 3pm"),
            ("The budget allocation is $2.4 million.", "$2.4 million"),
            ("The lead researcher is Dr. Elena Vasquez.", "Dr. Elena Vasquez"),
            ("The server password is xK9#mP2$vL.", "xK9#mP2$vL"),
            ("The launch date has been moved to October 28th.", "October 28th"),
            ("The quantum coherence time was 47 microseconds.", "47 microseconds"),
            ("The treaty was signed in Reykjavik in 1986.", "Reykjavik in 1986"),
            ("The species was first classified by Linnaeus in 1758.", "Linnaeus in 1758"),
            ("The satellite orbits at an altitude of 35,786 km.", "35,786 km"),
        ]

        task_idx = 0
        for ctx_size in context_sizes:
            target_chars = ctx_size * CHARS_PER_TOKEN

            for i in range(tasks_per_size):
                needle_text, needle_answer = needles[i % len(needles)]

                # Generate haystack
                haystack = SNIAHLoader._generate_haystack(rng, target_chars)

                # Insert needle at random position
                insert_pos = rng.randint(0, len(haystack))
                context = haystack[:insert_pos] + f"\n{needle_text}\n" + haystack[insert_pos:]

                tasks.append(BenchmarkTask(
                    task_id=f"sniah_{ctx_size // 1000}k_{task_idx:03d}",
                    context=context,
                    query=f"What is hidden in this text? Look for: '{needle_text[:30]}...'",
                    ground_truth=needle_answer,
                    context_tokens=len(context) // CHARS_PER_TOKEN,
                    scoring_fn="exact_match",
                    metadata={
                        "target_context_size": ctx_size,
                        "needle_position_ratio": insert_pos / len(haystack),
                    },
                ))
                task_idx += 1

        return BenchmarkDataset(name="sniah", tasks=tasks)

    @staticmethod
    def _generate_haystack(rng: random.Random, target_chars: int) -> str:
        """Generate filler text for the haystack."""
        paragraphs = []
        total = 0
        topics = [
            "The history of maritime navigation spans thousands of years.",
            "Agricultural practices vary widely across different climates.",
            "The development of printing technology transformed information sharing.",
            "Mountain ecosystems support remarkably diverse biological communities.",
            "Urban planning in the 21st century faces unprecedented challenges.",
            "The study of deep-sea organisms reveals surprising adaptations.",
            "Archaeological discoveries continue to reshape our understanding of history.",
            "Renewable energy technologies are advancing at an accelerating pace.",
            "The cognitive development of children follows predictable stages.",
            "International trade agreements shape economic relationships globally.",
        ]

        while total < target_chars:
            topic = rng.choice(topics)
            # Generate a paragraph with random filler
            filler_words = [
                "however", "furthermore", "consequently", "nevertheless",
                "specifically", "particularly", "additionally", "meanwhile",
                "subsequently", "accordingly", "interestingly", "notably",
            ]
            sentences = [topic]
            for _ in range(rng.randint(3, 8)):
                connector = rng.choice(filler_words)
                padding = " ".join(
                    rng.choice(string.ascii_lowercase) * rng.randint(3, 10)
                    for _ in range(rng.randint(5, 15))
                )
                sentences.append(f"{connector.capitalize()}, {padding}.")
            paragraph = " ".join(sentences)
            paragraphs.append(paragraph)
            total += len(paragraph)

        return "\n\n".join(paragraphs)[:target_chars]
