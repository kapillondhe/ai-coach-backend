from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "AI Coach API"
    environment: str = "development"
    debug: bool = True

    cors_origins: str = "http://localhost:3000"

    mcp_server_url: str
    mcp_auth_token: str | None = None

    openrouter_api_key: str | None = None
    # Tier 1: main coach conversation — tool-calling + tone-sensitive (incl. injury/physio advice).
    openrouter_model: str = "google/gemini-2.5-flash"
    # Tier 2: cheap utility model for lightweight tasks (currently: conversation titling
    # in app.services.titling; also suited to future summarization/classification work).
    openrouter_utility_model: str = "z-ai/glm-5.3-flash"

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
