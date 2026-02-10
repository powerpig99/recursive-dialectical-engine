"""Example: Document QA with partition strategies.

Demonstrates how the RDE uses semantic and structural partitioning
to handle long documents efficiently. Different traces get different
views of the document based on their context strategy.

Requirements:
    - At least one API key set (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
    - pip install recursive-dialectical-engine[cloud]

Usage:
    python -m examples.long_document
"""

from __future__ import annotations

import asyncio

from rde.engine import DialecticalEngine
from rde.models import ModelConfig, RecursionBudget


LONG_DOCUMENT = """
# Climate Change Impact on Marine Ecosystems

## Introduction

Climate change represents one of the most significant threats to marine
ecosystems worldwide. Rising ocean temperatures, acidification, and
changing current patterns are fundamentally altering the conditions that
marine organisms have adapted to over millennia.

## Temperature Effects

Ocean temperatures have risen by an average of 0.13 degrees Celsius per
decade since the 1970s. This warming is not uniform: polar regions are
experiencing rates 2-3x the global average. The thermal expansion of
seawater contributes approximately 42% of observed sea level rise.

Key finding: Species are migrating poleward at approximately 70 km per
decade on average, though some species show rates exceeding 200 km/decade.

## Ocean Acidification

Atmospheric CO2 absorption has reduced ocean pH by 0.1 units since
pre-industrial times. This 30% increase in hydrogen ion concentration
particularly affects calcifying organisms: corals, mollusks, and certain
plankton species.

Critical threshold: pH below 7.95 causes dissolution of aragonite shells
in cold waters, threatening the base of polar food webs.

## Biodiversity Impacts

Coral reef bleaching events have increased from once per decade (1980s)
to nearly annual occurrences. The Great Barrier Reef has experienced
mass bleaching in 2016, 2017, 2020, 2022, and 2024.

Biodiversity loss cascades through trophic levels: loss of coral habitat
reduces reef fish populations by approximately 60%, affecting dependent
species including sharks, sea turtles, and seabirds.

## Conclusion

The interconnected nature of marine ecosystem threats demands coordinated
global action. Current trajectories suggest irreversible tipping points
within 20-30 years without significant emissions reductions.
"""


async def main() -> None:
    config = ModelConfig()
    budget = RecursionBudget(max_depth=2, max_total_calls=20)

    async with DialecticalEngine(config) as engine:
        result = await engine.run(
            prompt=(
                f"Based on the following document, what is the most critical "
                f"finding regarding marine ecosystem thresholds?\n\n{LONG_DOCUMENT}"
            ),
            use_orchestrator=False,
            budget=budget,
        )

    print(f"Resolution: {result.resolution}")
    print(f"Confidence: {result.confidence.value}")
    print(f"Latency: {result.total_latency_ms:.0f}ms")
    if result.shadows:
        print("Shadows:")
        for s in result.shadows:
            print(f"  - {s}")


if __name__ == "__main__":
    asyncio.run(main())
