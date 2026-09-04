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

Set `ANTHROPIC_API_KEY` in `.env` for the coach agent to actually run (see `.env.example`).
`MCP_SERVER_URL` (default `http://localhost:8100/mcp`) points the agent at the MCP server.

- API root: http://localhost:8000
- Health: http://localhost:8000/api/health
- Coach chat: `POST http://localhost:8000/api/coach/chat` with `{"message": "..."}`
- Docs: http://localhost:8000/docs

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
  api/routes/        one module per feature area
  core/config.py     env-driven settings (pydantic-settings)
  agents/            Pydantic AI agents (coach_agent.py uses the MCP server as a toolset)
tests/
```

The MCP server (fitness-coaching tools) lives in a separate repo:
https://github.com/kapillondhe/ai-coach-mcp-server
