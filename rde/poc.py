"""Phase 0: Proof of Concept — local 3-trace dialectical engine on MLX.

Usage:
    uv run python -m rde.poc "Your problem here"
    uv run python -m rde.poc  # runs default Monty Hall accidental test
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid

from .models import (
    ArbitrationResult,
    ConfidenceLevel,
    EngineResult,
    ModelConfig,
    TraceConfig,
    TraceResult,
)
from .normalizer import TraceNormalizer
from .prompts.arbiter import build_arbiter_messages
from .prompts.traces import DEFAULT_STRUCTURED_TRACES, build_trace_messages
from .providers.mlx_provider import MLXProvider
from .utils.extraction import extract_boxed, extract_json_block


MONTY_HALL_ACCIDENTAL = """\
You are on a game show with 3 doors. Behind one is a car; behind the others, goats.
1. You pick Door 1.
2. The host, Monty, walks toward the remaining doors.
3. He slips on a banana peel and accidentally crashes into Door 2, forcing it open.
4. Door 2 reveals a Goat.
5. Monty picks himself up and asks: "Do you want to switch to Door 3?"

Does switching to Door 3 increase your probability of winning, or does it remain 50/50?
Derive the probability mathematically based strictly on the "Accidental" condition.
Put your final answer in \\boxed{}."""


async def run_poc(prompt: str, model_path: str | None = None) -> EngineResult:
    """Run the PoC 3-trace dialectical engine on a local MLX model."""
    config = ModelConfig()
    if model_path:
        config.local_model_path = model_path

    provider = MLXProvider(config.local_model_path)
    normalizer = TraceNormalizer()
    start = time.perf_counter()

    # 1. Use default structured traces (skip orchestrator LLM call)
    trace_configs = [TraceConfig(**t) for t in DEFAULT_STRUCTURED_TRACES]

    # 2. Execute traces sequentially (MLX can't parallelize on same GPU)
    trace_results: list[TraceResult] = []
    for tc in trace_configs:
        trace_id = f"{tc.role}_{uuid.uuid4().hex[:8]}"
        messages = build_trace_messages(tc.system_prompt, prompt)

        print(f"\n{'='*60}")
        print(f"Running trace: {tc.role} (temp={tc.temperature})")
        print(f"{'='*60}")

        try:
            response = await provider.complete(
                messages=messages,
                model="local",
                temperature=tc.temperature,
            )
            extracted = extract_boxed(response.content)
            result = TraceResult(
                trace_id=trace_id,
                role=tc.role,
                model_used=response.model,
                raw_output=response.content,
                extracted_answer=extracted,
                latency_ms=response.latency_ms,
            )
            print(f"Output ({response.latency_ms:.0f}ms):\n{response.content[:500]}")
            if extracted:
                print(f"\nExtracted answer: \\boxed{{{extracted}}}")
        except Exception as e:
            result = TraceResult(
                trace_id=trace_id,
                role=tc.role,
                model_used="local",
                raw_output="",
                error=str(e),
            )
            print(f"ERROR: {e}")

        trace_results.append(result)

    # 3. Normalize and check consensus
    successful = [r for r in trace_results if r.error is None]
    if not successful:
        return EngineResult(
            resolution="All traces failed",
            confidence=ConfidenceLevel.UNRESOLVED,
            shadows=["No traces produced output"],
            trace_results=trace_results,
            total_latency_ms=(time.perf_counter() - start) * 1000,
        )

    normalized = [normalizer.normalize(r) for r in successful]

    if normalizer.check_consensus(normalized):
        elapsed = (time.perf_counter() - start) * 1000
        print(f"\n{'='*60}")
        print(f"CONSENSUS reached: {normalized[0].conclusion}")
        print(f"{'='*60}")
        return EngineResult(
            resolution=normalized[0].conclusion,
            confidence=ConfidenceLevel.NECESSARY,
            causal_chain="All traces independently reached the same conclusion",
            trace_results=trace_results,
            total_latency_ms=elapsed,
            consensus_reached=True,
        )

    # 4. Arbiter resolves disagreement
    print(f"\n{'='*60}")
    print("DISAGREEMENT detected — invoking Arbiter")
    print(f"{'='*60}")

    trace_summaries = [
        {
            "role": t.role,
            "model": t.model_family,
            "conclusion": t.conclusion,
            "reasoning": "\n".join(t.reasoning_chain) if t.reasoning_chain else t.raw_output[:2000],
        }
        for t in normalized
    ]
    arbiter_messages = build_arbiter_messages(trace_summaries, prompt)

    try:
        arbiter_response = await provider.complete(
            messages=arbiter_messages,
            model="local",
            temperature=0.1,
        )
        print(f"\nArbiter output ({arbiter_response.latency_ms:.0f}ms):")
        print(arbiter_response.content[:1000])

        # Try to parse JSON from arbiter
        json_str = extract_json_block(arbiter_response.content)
        if json_str:
            parsed = json.loads(json_str)
            arbitration = ArbitrationResult.model_validate(parsed)
            arbitration.raw_output = arbiter_response.content
        else:
            # Fallback: use raw output as resolution
            arbitration = ArbitrationResult(
                resolution=arbiter_response.content[:500],
                causal_chain="Arbiter did not produce valid JSON",
                confidence=ConfidenceLevel.CONTINGENT,
                shadows=["Arbiter output could not be parsed as structured JSON"],
                raw_output=arbiter_response.content,
            )
    except Exception as e:
        # Fallback: pick trace with longest reasoning chain
        best = max(normalized, key=lambda t: len(t.reasoning_chain))
        arbitration = ArbitrationResult(
            resolution=best.conclusion,
            causal_chain=f"Arbiter failed ({e}); selected trace with most detailed reasoning",
            confidence=ConfidenceLevel.CONTINGENT,
            traces_adopted=[best.role],
            shadows=["Arbiter LLM call failed; resolution is best-effort fallback"],
            raw_output=str(e),
        )

    elapsed = (time.perf_counter() - start) * 1000
    return EngineResult(
        resolution=arbitration.resolution,
        confidence=arbitration.confidence,
        shadows=arbitration.shadows,
        causal_chain=arbitration.causal_chain,
        trace_results=trace_results,
        arbitration=arbitration,
        total_latency_ms=elapsed,
    )


def print_result(result: EngineResult) -> None:
    """Pretty-print the engine result."""
    print(f"\n{'='*60}")
    print(f"RESOLUTION: {result.resolution}")
    print(f"CONFIDENCE: {result.confidence.value}")
    if result.consensus_reached:
        print(f"(All {len(result.trace_results)} traces reached consensus)")
    print(f"{'='*60}")

    if result.causal_chain:
        print(f"\nCAUSAL CHAIN:\n{result.causal_chain}")

    if result.shadows:
        print("\nSHADOWS (what this resolution necessarily obscures):")
        for s in result.shadows:
            print(f"  - {s}")

    print(f"\nTRACES ({len(result.trace_results)}):")
    for t in result.trace_results:
        status = "OK" if not t.error else f"ERROR: {t.error}"
        answer = t.extracted_answer or "(no boxed answer)"
        print(f"  {t.role}: {answer} [{status}, {t.latency_ms:.0f}ms]")

    print(f"\nTotal latency: {result.total_latency_ms:.0f}ms")


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else MONTY_HALL_ACCIDENTAL
    if prompt == "--help":
        print(__doc__)
        return

    print(f"Prompt:\n{prompt}\n")
    result = asyncio.run(run_poc(prompt))
    print_result(result)


if __name__ == "__main__":
    main()
