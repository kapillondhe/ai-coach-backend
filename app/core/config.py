from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "AI Coach API"
    environment: str = "development"
    debug: bool = True

    # Comma-separated list of origins allowed to call the API.
    cors_origins: str = "http://localhost:3000"

    # MCP server (https://github.com/kapillondhe/ai-coach-mcp-server) that the coach agent calls as a tool provider.
    mcp_server_url: str
    mcp_auth_token: str | None = None

    openrouter_api_key: str | None = None
    openrouter_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"

    phoenix_api_key: str | None = None
    phoenix_collector_endpoint: str = "https://app.phoenix.arize.com"
    phoenix_project_name: str = "ai-coach"
    otel_service_name: str = "ai-coach-backend"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
