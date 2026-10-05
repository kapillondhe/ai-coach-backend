import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents.coach_agent import _base_toolset
from app.api.router import api_router
from app.core.config import get_settings
from app.core.db import close_engine, warm_up
from app.core.telemetry import setup_telemetry
from app.services import coros_mcp

settings = get_settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start the app even if our MCP server is unreachable.

    Opening `_base_toolset()` here used to be a hard prerequisite for startup: if the
    MCP server was down, the `async with` raised out of `lifespan`, FastAPI never
    finished starting, and the whole backend (every route, not just coach chat) went
    down with it. `MCPToolset` connects lazily per `async with self` anyway (see
    `list_tools`/`get_tools` in pydantic_ai.mcp), so pre-warming it is an optimization,
    not a requirement — a failed attempt here just means the first real chat request
    pays the connection cost (and fails with the existing 502 "temporarily unavailable"
    in app/api/routes/coach.py if the server is still down), instead of taking down
    health checks, auth, profile, dashboard, etc.
    """
    toolset = _base_toolset()
    mcp_entered = False
    try:
        await toolset.__aenter__()
        mcp_entered = True
    except Exception:
        logger.exception("Failed to connect to the MCP server at startup; continuing without it")

    await warm_up()
    try:
        yield
    finally:
        if mcp_entered:
            await toolset.__aexit__(None, None, None)
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
