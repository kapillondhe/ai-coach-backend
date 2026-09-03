# AI Coach — Backend (FastAPI)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Run

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

- API root: http://localhost:8000
- Health: http://localhost:8000/api/health
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
tests/
```
