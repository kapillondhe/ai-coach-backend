"""Test-wide settings isolation.

Tests must never read the developer's `.env` (real DB, OpenRouter, Phoenix, Supabase)
and must run on a clean checkout. This runs before any test module imports `app.*`.
"""

import os

import pytest

from app.core.config import Settings, get_settings

Settings.model_config["env_file"] = None

# Required settings get dummy values; nothing in the suite may reach these for real.
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@127.0.0.1:1/test")
os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")

get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _reset_chat_rate_limiter():
    """The chat limiter is process-global; don't let request counts leak between tests."""
    from app.core import rate_limit

    rate_limit.chat_limiter.reset()
    yield
    rate_limit.chat_limiter.reset()


@pytest.fixture(autouse=True)
def _stub_playbook_router(request):
    """Chat routes call the playbook router (a real Jev request) every turn; default it
    to "no playbooks" so route tests stay offline. test_playbooks.py tests the real one.

    Uses mock.patch, not `monkeypatch`: requesting monkeypatch from an autouse conftest
    fixture would reorder its teardown relative to other modules' own fixtures.
    """
    if request.module.__name__.endswith("test_playbooks"):
        yield
        return
    from unittest.mock import AsyncMock, patch

    from app.services import playbooks

    with patch.object(playbooks, "select_playbooks", AsyncMock(return_value=[])):
        yield
