# AI Coach — Backend (FastAPI)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Run

Two processes: the [MCP server](https://github.com/kapillondhe/ai-coach-mcp-server)
(fitness-coaching tools, its own repo) and this FastAPI backend, which calls it over
HTTP through a Pydantic AI agent.

```bash
source .venv/bin/activate

# terminal 1 - MCP server (streamable HTTP on :8100); see the ai-coach-mcp-server repo
python -m mcp_server

# terminal 2 - backend (calls the MCP server via app/agents/coach_agent.py)
uvicorn app.main:app --reload --port 8000
```

Set `OPENROUTER_API_KEY` in `.env` for the coach agent to actually run (see `.env.example`;
`OPENROUTER_MODEL` defaults to `nvidia/nemotron-3.5-lightning:free`).
`MCP_SERVER_URL` (default `http://localhost:8100/mcp`) points the agent at the MCP server.

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
  agents/            Pydantic AI agents (coach_agent.py uses the MCP server as a toolset)
tests/
```

The MCP server (fitness-coaching tools) lives in a separate repo:
https://github.com/kapillondhe/ai-coach-mcp-server
