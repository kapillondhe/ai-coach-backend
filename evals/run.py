"""Run the coach-agent eval suite against a real model + the live MCP server.

See `evals/README.md` for what's covered and why this lives outside `pytest`.

Usage (from `ai-coach-backend/`, with the venv active and the MCP server running):

    python -m evals.run
    python -m evals.run --case "Off-topic: coding help"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from pydantic_ai import Agent

from app.agents.coach_agent import CoachReply, get_coach_agent
from app.agents.models import get_utility_model
from evals.cases import all_cases
from evals.evaluators import DefersToMedicalCare, NoSystemPromptLeak

logging.basicConfig(level=logging.WARNING)


def _configure_tracing() -> None:
    """Register an in-process TracerProvider so span-based evaluators (ToolCorrectness)
    have a span tree to read.

    A plain SDK `TracerProvider` with no exporter attached — `pydantic_evals`'
    `context_subtree()` hooks itself a span processor directly, so nothing needs to be
    exported anywhere for this to work. Independent of `app.core.telemetry`'s Phoenix
    exporter, which stays off unless `PHOENIX_API_KEY` is set.
    """
    from opentelemetry import trace

    provider = TracerProvider(resource=Resource.create({"service.name": "ai-coach-evals"}))
    trace.set_tracer_provider(provider)
    Agent.instrument_all()


async def _run_coach_turn(message: str) -> CoachReply:
    """The task function: one fresh anonymous turn through the real coach agent.

    Anonymous (`user_id=None`) so no DB/Supabase/COROS setup is needed — matches how
    `app/api/routes/coach.py` builds the agent, minus the auth dependency resolution.
    """
    agent = await get_coach_agent(user_id=None)
    result = await agent.run(message)
    return result.output


def _build_dataset():
    from pydantic_evals import Dataset

    return Dataset(
        name="coach_agent",
        cases=all_cases,
        evaluators=[DefersToMedicalCare(), NoSystemPromptLeak()],
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="Run only the case with this exact name.")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Max concurrent cases (default 4 — keep modest, OpenRouter/MCP rate limits apply).",
    )
    args = parser.parse_args()

    _configure_tracing()

    from pydantic_evals.evaluators.llm_as_a_judge import set_default_judge_model

    # Judge with the cheap tier-2 model, not the pricier tier-1 coach model.
    set_default_judge_model(get_utility_model())

    dataset = _build_dataset()
    if args.case:
        matching = [c for c in dataset.cases if c.name == args.case]
        if not matching:
            print(f"No case named {args.case!r}. Available: {[c.name for c in dataset.cases]}")
            return 1
        dataset.cases = matching

    report = await dataset.evaluate(_run_coach_turn, max_concurrency=args.max_concurrency)
    report.print(include_input=True, include_output=True, include_reasons=True)

    any_failed = any(not result.value for case in report.cases for result in case.assertions.values()) or bool(
        report.failures
    )
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
