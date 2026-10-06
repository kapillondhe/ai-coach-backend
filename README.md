# AI Coach — Backend (FastAPI)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Run

One process: this FastAPI backend, whose Pydantic AI coach agent runs the
fitness-coaching tools in-process (`app/tools/`). The knowledge-base search tool needs
a Qdrant instance populated from `knowledge_base/`.

```bash
source .venv/bin/activate
docker compose up -d                     # local Qdrant (:6333)
python -m scripts.ingest_knowledge_base  # first time, and after editing knowledge_base/*.md
uvicorn app.main:app --reload --port 8000
```

Set `OPENROUTER_API_KEY` in `.env` for the coach agent to actually run (see `.env.example`).
`OPENROUTER_MODEL` (default `z-ai/glm-5.3-flash`) runs the main coach conversation;
`OPENROUTER_UTILITY_MODEL` is a cheaper model for lightweight tasks like conversation
titling, and `OPENROUTER_JEV_MODEL` is the yes/no decision model behind the off-topic and
memory-save guardrails. All three are built in `app/agents/models.py`.
Chat is rate-limited per user (or per client IP when anonymous) via
`CHAT_RATE_LIMIT_PER_MINUTE` (default 20, `0` disables). The limiter is in-process, so
limits apply per worker.
`QDRANT_URL` (default `http://localhost:6333`, plus `QDRANT_API_KEY` for Qdrant Cloud)
points the knowledge-base tool at Qdrant; if it's unreachable the tool returns no
results and chat carries on without grounding. Production uses Qdrant Cloud, which
computes embeddings server-side (Cloud Inference); locally the same model runs via
fastembed. To (re-)load production's knowledge base:
`QDRANT_URL=<cloud url> python -m scripts.ingest_knowledge_base` with `QDRANT_API_KEY` in `.env`.

- API root: http://localhost:8000
- Health: http://localhost:8000/api/health
- Coach chat: `POST http://localhost:8000/api/coach/chat` with `{"message": "...", "session_id": "..."}`
  (plus `conversation_id` for a signed-in user's saved conversation, or `history` when anonymous)
- Coach chat (streaming SSE): `POST http://localhost:8000/api/coach/chat/stream`, same body
- Signed-in routes (Supabase JWT in `Authorization: Bearer ...`): `/api/profile`,
  `/api/conversations`, `/api/memories`, `/api/dashboard/summary`, `/api/integrations/coros/*`
- Docs: http://localhost:8000/docs

Tracing to Arize Phoenix is optional — set `PHOENIX_API_KEY` in `.env` to enable it (see
`app/core/telemetry.py`); left unset it no-ops.

## Test and lint

```bash
source .venv/bin/activate
pytest                   # no .env, DB or network needed (tests/conftest.py supplies dummy settings)
ruff check . && ruff format --check .
pyright                  # basic mode, configured in pyproject.toml
alembic check            # models in app/core/models.py must match the migrated schema
```

CI (`.github/workflows/deploy.yml`) runs ruff, pyright and pytest, and deploys only when they pass.

Dependencies: production installs from `pyproject.toml`; `requirements.txt` is the local
dev install and must keep identical pins (dev/test/evals tools live in the `dev` extra).

## Layout

```
app/
  main.py            FastAPI app, CORS, router wiring
  api/router.py      aggregates all route modules under /api
  api/routes/        one module per feature area: health, coach, conversations, memories,
                     profile, dashboard, integrations (COROS OAuth), sync (internal cron trigger)
  core/config.py     env-driven settings (pydantic-settings)
  core/auth.py       Supabase JWT verification (anonymous when no/invalid token)
  core/db.py         async SQLAlchemy engine/session
  core/models.py     ORM models (Alembic target_metadata; must match migrations/)
  core/rate_limit.py in-process chat rate limiter
  core/telemetry.py  OpenTelemetry export to Arize Phoenix (no-ops without PHOENIX_API_KEY)
  agents/models.py   OpenRouter model factories (coach, utility, Jev decision model)
  agents/coach_agent.py  the coach Agent: prompt, toolsets (coaching + COROS), output safety
  tools/             in-process coaching tools (nutrition, zones, knowledge-base search)
  services/          business logic: conversations, memory, profile, dashboard, COROS
                     OAuth/MCP pool/sync, guardrails (safety, scope, memory_guardrail),
                     vector_store.py (Qdrant client for the knowledge base)
migrations/          Alembic migrations
knowledge_base/      curated coaching content (*.md), ingested into Qdrant
scripts/             ingest_knowledge_base.py
evals/               LLM-output evals against the real model (not run by pytest)
tests/
```
