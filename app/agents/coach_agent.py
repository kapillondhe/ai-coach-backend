import logging
from functools import lru_cache

from fastapi import Depends
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.toolsets import AbstractToolset

from app.core.auth import get_current_user_id
from app.core.config import get_settings
from app.services import coros_oauth

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an encouraging, knowledgeable fitness coach. Use the available "
    "tools to log and look up a user's workouts before giving advice."
)


@lru_cache
def _get_model() -> OpenRouterModel:
    settings = get_settings()
    return OpenRouterModel(
        settings.openrouter_model,
        provider=OpenRouterProvider(api_key=settings.openrouter_api_key),
    )


def _base_toolset() -> MCPToolset:
    settings = get_settings()
    return MCPToolset(
        client=settings.mcp_server_url,
        headers={"Authorization": f"Bearer {settings.mcp_auth_token}"} if settings.mcp_auth_token else None,
    )


async def _build_toolsets(user_id: str | None) -> list[AbstractToolset]:
    toolsets: list[AbstractToolset] = [_base_toolset()]

    if user_id is not None:
        try:
            access_token = await coros_oauth.get_access_token(user_id)
        except Exception:
            logger.exception("Failed to resolve COROS access token for user %s", user_id)
            access_token = None

        if access_token:
            settings = get_settings()
            toolsets.append(
                MCPToolset(
                    client=settings.coros_mcp_server_url,
                    headers={"Authorization": f"Bearer {access_token}"},
                )
            )

    return toolsets


async def get_coach_agent(user_id: str | None = Depends(get_current_user_id)) -> Agent:
    """Build a coach Agent for this request; not cached since attached toolsets depend on the user."""
    return Agent(
        model=_get_model(),
        toolsets=await _build_toolsets(user_id),
        system_prompt=SYSTEM_PROMPT,
    )
