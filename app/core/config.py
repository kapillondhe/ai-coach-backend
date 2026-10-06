from functools import lru_cache
from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "AI Coach API"
    environment: str = "development"
    debug: bool = False

    cors_origins: str = "http://localhost:3000"

    # Qdrant backing the knowledge-base search tool (app/services/vector_store.py).
    # Local dev default assumes `docker compose up -d`; production points at Qdrant
    # Cloud, which also needs qdrant_api_key. Populate with scripts/ingest_knowledge_base.py.
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "coaching_kb"
    # Embed server-side via Qdrant Cloud Inference instead of locally with fastembed.
    # Unset = auto: on for a *.cloud.qdrant.io URL, off for local/self-hosted Qdrant.
    qdrant_cloud_inference_override: bool | None = Field(default=None, alias="QDRANT_CLOUD_INFERENCE")

    openrouter_api_key: str | None = None
    # Tier 1: main coach conversation — tool-calling + tone-sensitive (incl. injury/physio advice).
    openrouter_model: str = "z-ai/glm-5.3-flash"
    # Tier 2: cheap utility model for lightweight tasks (currently: conversation titling
    # in app.services.titling; also suited to future summarization/classification work).
    openrouter_utility_model: str = "z-ai/glm-5.3-flash"
    # Decision model for typed yes/no classification (currently: the scope guardrail in
    # app.services.scope) — billed through the same OpenRouter account/key, called via
    # pydantic-ai's SystemOneModel (a typed decision, not a chat completion).
    openrouter_jev_model: str = "typesafe/jev-1.13"

    # Max requests per minute to /api/coach/chat and /chat/stream, per signed-in user
    # (else per client IP); 0 disables. Enforced in-process, so each worker counts
    # separately (app/core/rate_limit.py).
    chat_rate_limit_per_minute: int = 20

    phoenix_api_key: str | None = None
    phoenix_collector_endpoint: str = "https://app.phoenix.arize.com"
    phoenix_project_name: str = "ai-coach"
    otel_service_name: str = "ai-coach-backend"

    database_url: str

    token_encryption_key: str | None = None

    coros_mcp_server_url: str = "https://mcp.coros.com/mcp"
    coros_redirect_uri: str = "http://localhost:8000/api/integrations/coros/callback"

    internal_sync_secret: str | None = None

    frontend_url: str = "http://localhost:3000"

    supabase_url: str | None = None
    supabase_jwt_audience: str = "authenticated"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def qdrant_cloud_inference(self) -> bool:
        if self.qdrant_cloud_inference_override is not None:
            return self.qdrant_cloud_inference_override
        host = urlparse(self.qdrant_url).hostname or ""
        return host.endswith(".cloud.qdrant.io")


@lru_cache
def get_settings() -> Settings:
    # Required fields (database_url) come from the environment / .env, which pyright can't see.
    return Settings()  # pyright: ignore[reportCallIssue]
