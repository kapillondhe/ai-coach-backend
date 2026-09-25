from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.api.routes import dashboard as dashboard_routes
from app.core.auth import require_user_id
from app.main import app
from app.services.dashboard import DashboardSummary, SparklinePoint, TrendCard


def test_dashboard_summary_requires_sign_in():
    app.dependency_overrides.clear()
    client = TestClient(app)
    assert client.get("/api/dashboard/summary").status_code == 401


def test_dashboard_summary_not_connected(monkeypatch):
    app.dependency_overrides[require_user_id] = lambda: "user-123"
    monkeypatch.setattr(
        dashboard_routes.dashboard_service,
        "get_dashboard_summary",
        AsyncMock(return_value=DashboardSummary(connected=False)),
    )

    client = TestClient(app)
    response = client.get("/api/dashboard/summary")

    assert response.status_code == 200
    assert response.json() == {
        "connected": False,
        "syncing": False,
        "trends": [],
    }
    app.dependency_overrides.clear()


def test_dashboard_summary_syncing(monkeypatch):
    app.dependency_overrides[require_user_id] = lambda: "user-123"
    monkeypatch.setattr(
        dashboard_routes.dashboard_service,
        "get_dashboard_summary",
        AsyncMock(return_value=DashboardSummary(connected=True, syncing=True)),
    )

    client = TestClient(app)
    response = client.get("/api/dashboard/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["connected"] is True
    assert body["syncing"] is True
    app.dependency_overrides.clear()


def test_dashboard_summary_connected_with_data(monkeypatch):
    app.dependency_overrides[require_user_id] = lambda: "user-123"
    monkeypatch.setattr(
        dashboard_routes.dashboard_service,
        "get_dashboard_summary",
        AsyncMock(
            return_value=DashboardSummary(
                connected=True,
                syncing=False,
                trends=[
                    TrendCard(
                        id="resting_hr",
                        label="Resting heart rate",
                        sparkline=[SparklinePoint(date="2026-09-24", value=53.0)],
                        takeaway="Stable.",
                        latest_value=53.0,
                    )
                ],
            )
        ),
    )

    client = TestClient(app)
    response = client.get("/api/dashboard/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["connected"] is True
    assert "discipline_load" not in body
    assert body["trends"][0]["id"] == "resting_hr"
    app.dependency_overrides.clear()
