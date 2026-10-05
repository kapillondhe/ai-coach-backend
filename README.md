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
`OPENROUTER_MODEL` defaults to `google/gemini-2.5-flash` for the main coach conversation;
`OPENROUTER_UTILITY_MODEL` defaults to `z-ai/glm-5.3-flash`, a cheaper model used for
lightweight tasks like conversation titling.
`QDRANT_URL` (default `http://localhost:6333`, plus `QDRANT_API_KEY` for Qdrant Cloud)
points the knowledge-base tool at Qdrant; if it's unreachable the tool returns no
results and chat carries on without grounding. Production uses Qdrant Cloud, which
computes embeddings server-side (Cloud Inference); locally the same model runs via
fastembed. To (re-)load production's knowledge base:
`QDRANT_URL=<cloud url> python -m scripts.ingest_knowledge_base` with `QDRANT_API_KEY` in `.env`.

- API root: http://localhost:8000
- Health: http://localhost:8000/api/health
- Coach chat: `POST http://localhost:8000/api/coach/chat` with `{"message": "...", "session_id": "..."}`
- Coach chat (streaming SSE): `POST http://localhost:8000/api/coach/chat/stream`, same body
- Docs: http://localhost:8000/docs

Tracing to Arize Phoenix is optional — set `PHOENIX_API_KEY` in `.env` to enable it (see
`app/core/telemetry.py`); left unset it no-ops.

## Test

```bash
source .venv/bin/activate
pytest
```

## Layout

```
app/
  main.py            FastAPI app, CORS, router wiring
  api/router.py      aggregates all route modules under /api
  api/routes/        one module per feature area (health, coach)
  core/config.py     env-driven settings (pydantic-settings)
  core/telemetry.py  OpenTelemetry export to Arize Phoenix (no-ops without PHOENIX_API_KEY)
  agents/            Pydantic AI agents (coach_agent.py wires in the coaching toolset + COROS)
  tools/             in-process coaching tools (nutrition, zones, knowledge-base search)
  services/          business logic incl. vector_store.py (Qdrant client for the knowledge base)
knowledge_base/      curated coaching content (*.md), ingested into Qdrant
scripts/             ingest_knowledge_base.py
tests/
```
