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

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
