# AGENTS.md

Guidance for coding agents working in this repo. Setup, run commands and the module
layout are in README.md.

## Gates (CI runs all of them; run them before calling a change done)

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest -q
uv run alembic check   # needs DATABASE_URL; read-only, safe against a real DB
```

Never run `alembic upgrade`/`downgrade`/`stamp` or any DDL/DML against a DB you did not
create for the purpose.

## Dependencies

Declare them in `pyproject.toml` only, then `uv lock` and regenerate `requirements.txt`
with `uv export --frozen --extra dev --no-hashes --no-emit-project -o requirements.txt`.
Never hand-edit `requirements.txt`. Prod (FastAPI Cloud) installs from `uv.lock` on
Python 3.14.

## Tests

- The suite runs with no `.env`, DB or network: `tests/conftest.py` sets dummy settings
  before any `app.*` import. Keep it that way.
- Route tests fake the agent via `app.dependency_overrides[get_coach_agent]` and stub
  `scope_service.is_off_topic`; otherwise they make real, paid model calls.
- Service tests fake the DB with `tests/fakes.py` (`FakeSession`, `ScriptedSession`),
  dispatching on statement type. A new statement type needs a new branch in the fake.
- Every bug fix gets a regression test that fails without the fix.
- Process-global state (rate limiter, caches, `lru_cache`d settings) must be reset by a
  fixture, or results leak between tests.

## Code conventions

- Model factories live in `app/agents/models.py` (coach, utility, Jev). Import them from
  there, not from `coach_agent.py`, to avoid import cycles.
- Coach tools: raise `ModelRetry` for bad arguments so the model can correct itself; a
  plain exception fails the whole chat turn.
- Optional external dependencies (COROS, guardrail classifiers, playbook router, profile
  and memory lookups) fail open and log: their trouble must never break core chat. Safety
  checks on the model's own reply fail closed.
- Never make app startup (`lifespan`) depend on an external service being up.
- `app/core/models.py` must match the migrations; `alembic check` must report no changes.
- Changing a client-visible contract (status codes, request limits, response shapes, SSE
  events) needs a matching frontend change; a response header the browser must read has
  to be listed in CORS `expose_headers` in `app/main.py`.
- Docstrings and comments explain why, not change history.

## Don't

- Commit `.env` or print secrets. `CLAUDE.md` is gitignored (local-only notes).
- Run evals (`python -m evals.run`) or other live-model scripts without asking: they cost
  money.
