import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic_ai import Agent

from app.agents.coach_agent import _get_model
from app.api.router import api_router
from app.core.config import get_settings
from app.core.db import close_engine, warm_up
from app.core.telemetry import setup_telemetry
from app.services import coros_mcp
from app.services import scope as scope_service
from app.tools.knowledge_base import search_knowledge_base

settings = get_settings()
logger = logging.getLogger(__name__)


async def _warm_knowledge_base() -> None:
    """Open the Qdrant connection (and, in local dev, load fastembed's model) off the request path.

    In production embeddings run on Qdrant Cloud, so this is just a connection warm-up;
    locally fastembed loads its ONNX model on first use, which would otherwise land on
    the first chat that searches the knowledge base. Runs in the background and never
    blocks or fails startup — search_knowledge_base already swallows its own errors.
    """
    await search_knowledge_base("warm-up", top_k=1)


async def _warm_coach_models() -> None:
    """Open the OpenRouter connections for the tier-1 coach model and the Jev scope

    classifier off the request path. `OpenRouterModel`/`SystemOneModel` build their
    own httpx client (and OpenRouter does its own cold routing lookup) lazily on first
    use, not at construction — measured at ~13s extra on whichever user's chat message
    happens to be first after a cold start/deploy, dropping to ~2s once warm. A tiny
    real completion on each model pays that cost here instead. Runs in the background
    and never blocks or fails startup. Reuses `scope_service.is_off_topic` (rather than a
    bare Agent call) for the Jev warm-up since it already fails open and is the exact
    shape (instructions + output_type) the real guardrail call needs.
    """
    try:
        await Agent(model=_get_model()).run("Reply with OK.")
    except Exception:
        logger.warning("Coach model warm-up failed; first chat request will connect lazily", exc_info=True)
    await scope_service.is_off_topic("warm-up")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await warm_up()
    # Started after warm_up() so a DB warm-up failure can't leave these tasks orphaned.
    kb_warmup = asyncio.create_task(_warm_knowledge_base())
    model_warmup = asyncio.create_task(_warm_coach_models())
    try:
        yield
    finally:
        for task in (kb_warmup, model_warmup):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await coros_mcp.close_all()
        await close_engine()


app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)
setup_telemetry(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": f"{settings.app_name} is running"}
