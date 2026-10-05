import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import get_settings
from app.core.db import close_engine, warm_up
from app.core.telemetry import setup_telemetry
from app.services import coros_mcp
from app.tools.knowledge_base import search_knowledge_base

settings = get_settings()


async def _warm_knowledge_base() -> None:
    """Open the Qdrant connection (and, in local dev, load fastembed's model) off the request path.

    In production embeddings run on Qdrant Cloud, so this is just a connection warm-up;
    locally fastembed loads its ONNX model on first use, which would otherwise land on
    the first chat that searches the knowledge base. Runs in the background and never
    blocks or fails startup — search_knowledge_base already swallows its own errors.
    """
    await search_knowledge_base("warm-up", top_k=1)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await warm_up()
    # Started after warm_up() so a DB warm-up failure can't leave this task orphaned.
    kb_warmup = asyncio.create_task(_warm_knowledge_base())
    try:
        yield
    finally:
        kb_warmup.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await kb_warmup
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
