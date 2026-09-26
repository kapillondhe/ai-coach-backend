import logging
from functools import lru_cache

from fastapi import Depends
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.toolsets import AbstractToolset

from app.core.auth import get_current_user_id
from app.core.config import get_settings
from app.services import coros_oauth
from app.services import memory as memory_service

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an encouraging, knowledgeable fitness coach. Use the available "
    "tools to log and look up a user's workouts before giving advice.\n\n"
    "After writing your reply, also fill in `suggestions`: 0 to 3 short "
    "(under ~8 words), first-person follow-up messages the user might "
    "plausibly send next, phrased as if the user were typing them (e.g. "
    "\"How much protein do I need?\"), directly relevant to what was just "
    "discussed. Leave it empty if nothing natural fits — don't force it."
)


class CoachReply(BaseModel):
    """Structured output for a coach turn: the reply plus optional follow-up suggestions."""

    reply: str
    suggestions: list[str] = Field(default_factory=list, max_length=3)

_MEMORY_TOOL_DESCRIPTION = (
    "Record a short, durable fact or preference the signed-in user just stated "
    "(e.g. an injury, a goal race, a dietary preference) so it's remembered in "
    "future conversations, not just this one. Only call this for things worth "
    "recalling later — not every message."
)


@lru_cache
def _get_model() -> OpenRouterModel:
    """Tier 1: main coach conversation model — tool-calling + tone-sensitive."""
    settings = get_settings()
    return OpenRouterModel(
        settings.openrouter_model,
        provider=OpenRouterProvider(api_key=settings.openrouter_api_key),
    )


@lru_cache
def get_utility_model() -> OpenRouterModel:
    """Tier 2: cheap utility model for lightweight tasks (summarization, classification,
    titling). Used by app.services.titling for conversation title generation."""
    settings = get_settings()
    return OpenRouterModel(
        settings.openrouter_utility_model,
        provider=OpenRouterProvider(api_key=settings.openrouter_api_key),
    )


@lru_cache
def _base_toolset() -> MCPToolset:
    """Our own MCP server's toolset — a single long-lived connection shared across requests.
    """
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


async def _build_system_prompt(user_id: str | None) -> str:
    if user_id is None:
        return SYSTEM_PROMPT

    try:
        memories = await memory_service.list_memories(user_id)
    except Exception:
        logger.exception("Failed to load stored memories for user %s", user_id)
        return SYSTEM_PROMPT

    if not memories:
        return SYSTEM_PROMPT

    bullets = "\n".join(f"- {m.content}" for m in memories)
    prompt = f"{SYSTEM_PROMPT}\n\nThings you remember about this user from past conversations:\n{bullets}"
    logger.info("Injecting %d stored memories into system prompt for user %s", len(memories), user_id)
    return prompt


async def get_coach_agent(user_id: str | None = Depends(get_current_user_id)) -> Agent[None, CoachReply]:
    """Build a coach Agent for this request; not cached since attached toolsets depend on the user."""
    agent = Agent(
        model=_get_model(),
        output_type=CoachReply,
        toolsets=await _build_toolsets(user_id),
        system_prompt=await _build_system_prompt(user_id),
    )

    if user_id is not None:

        @agent.tool_plain(description=_MEMORY_TOOL_DESCRIPTION)
        async def remember(fact: str) -> str:
            await memory_service.remember(user_id, fact)
            return "Noted."

    return agent
