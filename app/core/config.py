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
    openrouter_model: str = "z-ai/glm-5.3-flash"

    phoenix_api_key: str | None = None
    phoenix_collector_endpoint: str = "https://app.phoenix.arize.com"
    phoenix_project_name: str = "ai-coach"
    otel_service_name: str = "ai-coach-backend"

    database_url: str

    token_encryption_key: str | None = None


    coros_mcp_server_url: str = "https://mcp.coros.com/mcp"
    coros_redirect_uri: str = "http://localhost:8000/api/integrations/coros/callback"

    
    frontend_url: str = "http://localhost:3000"


    supabase_url: str | None = None


    supabase_jwt_audience: str = "authenticated"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
