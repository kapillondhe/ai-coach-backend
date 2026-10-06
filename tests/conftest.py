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
