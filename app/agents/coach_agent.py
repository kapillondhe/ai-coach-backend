from functools import lru_cache

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from app.core.config import get_settings

SYSTEM_PROMPT = (
    "You are an encouraging, knowledgeable fitness coach. Use the available "
    "tools to log and look up a user's workouts before giving advice."
)


@lru_cache
def get_coach_agent() -> Agent:
    settings = get_settings()

    model = OpenRouterModel(
        settings.openrouter_model,
        provider=OpenRouterProvider(api_key=settings.openrouter_api_key),
    )
    mcp_toolset = MCPToolset(
        client=settings.mcp_server_url,
        headers={"Authorization": f"Bearer {settings.mcp_auth_token}"} if settings.mcp_auth_token else None,
    )
    return Agent(model=model, toolsets=[mcp_toolset], system_prompt=SYSTEM_PROMPT)
