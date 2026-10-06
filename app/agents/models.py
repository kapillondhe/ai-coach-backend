"""Process-wide model singletons, one per tier.

Kept out of `coach_agent` so services that `coach_agent` itself imports
(`memory_guardrail`) can use a model without a circular import.
"""

from functools import lru_cache

from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.models.system_one import SystemOneModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.providers.system_one import SystemOneProvider

from app.core.config import get_settings


@lru_cache
def get_coach_model() -> OpenRouterModel:
    """Tier 1: main coach conversation model — tool-calling + tone-sensitive.

    Reasoning effort is pinned to the lowest setting: this model's OpenRouter
    endpoint requires reasoning (it 400s if asked to disable it entirely), but
    minimal effort still cuts most of the hidden chain-of-thought latency that
    piles up before every reply and every tool call.
    """
    settings = get_settings()
    return OpenRouterModel(
        settings.openrouter_model,
        provider=OpenRouterProvider(api_key=settings.openrouter_api_key),
        settings=OpenRouterModelSettings(openrouter_reasoning={"effort": "minimal"}),
    )


@lru_cache
def get_utility_model() -> OpenRouterModel:
    """Tier 2: cheap model for lightweight generation (titling, eval judging)."""
    settings = get_settings()
    return OpenRouterModel(
        settings.openrouter_utility_model,
        provider=OpenRouterProvider(api_key=settings.openrouter_api_key),
    )


@lru_cache
def get_jev_model() -> SystemOneModel:
    """Decision model (TypeSafe's Jev) for typed yes/no classification guardrails
    (`app.services.scope`, `app.services.memory_guardrail`).

    Routed through OpenRouter's `/v1/systemone`-compatible endpoint via `SystemOneModel`
    + `SystemOneProvider` (billed to the same OPENROUTER_API_KEY as every other model
    here) rather than `pydantic_ai.models.typesafe.TypeSafeModel`, which talks directly
    to TypeSafe's own API and needs a separate TYPESAFE_API_KEY/account.
    """
    settings = get_settings()
    provider = SystemOneProvider(base_url="https://openrouter.ai/api", api_key=settings.openrouter_api_key)
    return SystemOneModel(settings.openrouter_jev_model, provider=provider)
