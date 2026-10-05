# Coach agent evals

LLM-output evals for the coach agent (`app/agents/coach_agent.py`), separate from
`tests/` (which inject a fake agent and never call a real model). These evals run the
*real* `get_coach_agent(user_id=None)` against OpenRouter (with the in-process
coaching tools), and score replies for scope/safety/tool-use/quality. They're
slow, non-deterministic, and cost real tokens — run them deliberately, not on every
commit.

## What's covered

`cases.py` holds ~18 hand-written cases grouped by what they probe:

- **Scope**: off-topic requests (coding help, trivia, movies) get declined and
  redirected, not answered.
- **Tool use**: a request that gives enough info for a specific coaching tool
  (`calculate_protein_intake`, `calculate_heart_rate_zones`, `calculate_power_zones`,
  `calculate_swim_pace_zones`, `search_knowledge_base`) actually triggers that tool.
- **Safety**: symptom phrasing that's *not* caught by `app/services/safety.py`'s
  narrow emergency-regex pre-check (that pre-check is covered by
  `tests/test_safety.py` and short-circuits before the agent ever runs) but that the
  system prompt + output validator should still catch and defer to medical care for.
- **Prompt injection**: direct and embedded attempts to extract the system prompt or
  override the coach's role.
- **Suggestion-chip safety**: slot-filling replies (asking for weight/pace/etc.) must
  not produce a suggestion chip that invents a specific personal number, per
  `ai-coach-backend-patterns` skill's documented pitfall.
- **General quality**: a couple of open-ended coaching questions scored by an LLM
  judge (`GEval`) for actionable, sport-appropriate advice, since most cases above use
  cheap deterministic or tool-trajectory checks instead.

`evaluators.py` has two small custom `Evaluator`s that reuse the existing
`app/services/safety.py` checks and a regex for digit-bearing suggestion chips.
Everything else is a built-in from `pydantic_evals.evaluators`
(`ToolCorrectness`, `LLMJudge`, `GEval`).

## Running

```bash
source .venv/bin/activate
python -m evals.run                  # pretty-prints a report table to stdout
python -m evals.run --case "Off-topic: coding help"   # one case by name
```

Requires `OPENROUTER_API_KEY` in `.env` (same as normal backend operation) and a
populated local Qdrant (`docker compose up -d` + `python -m scripts.ingest_knowledge_base`)
for the knowledge-base cases, exactly like running the
backend for real — `evals/run.py` builds the agent via the same
`app.agents.coach_agent.get_coach_agent` the FastAPI route uses, with no mocking.

The `LLMJudge`/`GEval` cases use the tier-2 utility model
(`app.agents.coach_agent.get_utility_model()`) as the judge, not the pricier tier-1
coach model, to keep judging cheap — see `set_default_judge_model` in `run.py`.

## Tool-trajectory evaluators need tracing enabled

`ToolCorrectness` reads tool calls from an in-process OpenTelemetry span tree, which
only gets recorded if a `TracerProvider` is registered and `Agent.instrument_all()`
has been called — both happen once, process-wide, at the top of `run.py`. This is
independent of (and doesn't send anything to) the Arize Phoenix exporter in
`app/core/telemetry.py`, which stays off unless `PHOENIX_API_KEY` is set.

## Why this isn't in `tests/`

Pytest's suite is meant to be fast, deterministic, and run on every change (CI,
pre-commit) — these evals are the opposite on all three counts: real model calls,
non-deterministic grading, real latency/cost. Keeping them in `evals/` (not collected
by `pytest`'s `testpaths = ["tests"]`) makes that boundary explicit.
