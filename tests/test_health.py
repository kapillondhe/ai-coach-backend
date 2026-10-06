from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "environment": "development"}


def test_debug_defaults_to_off_and_model_matches_env_example(monkeypatch) -> None:
    from app.core.config import Settings

    monkeypatch.delenv("DEBUG", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    settings = Settings()

    assert settings.debug is False
    assert settings.openrouter_model == "z-ai/glm-5.3-flash"


def test_openapi_documents_typed_response_models() -> None:
    schemas = client.get("/openapi.json").json()["components"]["schemas"]

    for name in (
        "HealthResponse",
        "ChatResponse",
        "ChatOpenerResponse",
        "ConversationDeletedResponse",
        "MemoryDeletedResponse",
        "CorosAuthorizationResponse",
        "CorosStatusResponse",
        "CorosDisconnectResponse",
        "SyncRunResponse",
    ):
        assert name in schemas, name
