"""app.agents.models: process-wide model singletons, importable without a cycle."""

import subprocess
import sys
from pathlib import Path

from app.agents import models


def test_model_factories_are_cached_singletons():
    assert models.get_coach_model() is models.get_coach_model()
    assert models.get_utility_model() is models.get_utility_model()
    assert models.get_jev_model() is models.get_jev_model()


def test_services_use_the_shared_model_module():
    from app.services import memory_guardrail, scope, titling

    assert memory_guardrail.get_jev_model is models.get_jev_model
    assert scope.get_jev_model is models.get_jev_model
    assert titling.get_utility_model is models.get_utility_model


def test_memory_guardrail_imports_cleanly_first():
    # Regression for the old circular import (coach_agent <-> memory_guardrail):
    # importing the guardrail first in a fresh interpreter must not fail.
    code = "import app.services.memory_guardrail, app.agents.coach_agent"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parents[1])
