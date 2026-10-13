"""Tests for app.services.playbooks: content loading and the Jev playbook router.

Mocks `Agent.run` instead of calling Jev, same convention as test_scope_service.py.
Real routing accuracy is measured with evals/playbook_routing.py (live calls).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from app.agents.coach_agent import SYSTEM_PROMPT
from app.services import playbooks

EXPECTED_ROUTABLE = {
    "intensity-and-zones",
    "training-week",
    "adjusting-the-plan",
    "race-execution",
    "fueling-and-hydration",
    "recovery-and-readiness",
    "pain-and-injury",
    "strength-and-mobility",
    "swim-and-open-water",
}


def _selection(*true_ids: str):
    model = playbooks._selection_type()
    return model(**{playbooks._field_name(p.id): p.id in true_ids for p in playbooks.routable_playbooks()})


def _run_result(output, confidence: dict[str, float] | None = None):
    response = ModelResponse(parts=[], provider_details={"confidence": confidence or {}})
    return SimpleNamespace(output=output, all_messages=lambda: [ModelRequest(parts=[]), response])


def test_loads_every_playbook_with_metadata():
    loaded = playbooks.load_playbooks()

    assert {p.id for p in playbooks.routable_playbooks()} == EXPECTED_ROUTABLE
    assert playbooks.CORE_ID in loaded
    for playbook in loaded.values():
        assert playbook.title
        assert len(playbook.description) > 30, playbook.id
        assert len(playbook.body.split()) > 150, playbook.id


def test_sources_section_is_not_sent_to_the_model():
    for playbook in playbooks.load_playbooks().values():
        assert "## Sources" not in playbook.body, playbook.id


def test_frontmatter_continuation_lines_join_into_one_description():
    description = playbooks.load_playbooks()["fueling-and-hydration"].description

    assert "\n" not in description
    assert description.startswith("What to eat and drink")
    assert description.endswith("under-fueling.")


def test_parse_rejects_id_that_does_not_match_filename(tmp_path):
    path = tmp_path / "wrong-name.md"
    path.write_text("---\nid: something-else\ntitle: T\ndescription: D\n---\n\nBody\n")

    with pytest.raises(ValueError, match="doesn't match its filename"):
        playbooks._parse_playbook(path)


def test_coaching_core_is_in_the_system_prompt():
    assert playbooks.coaching_core() in SYSTEM_PROMPT
    # Routed playbooks are not: they arrive per turn as run instructions.
    for playbook in playbooks.routable_playbooks():
        assert playbook.body not in SYSTEM_PROMPT, playbook.id


def test_router_question_per_playbook_carries_its_description():
    schema = playbooks._selection_type().model_json_schema()

    assert set(schema["properties"]) == {playbooks._field_name(pid) for pid in EXPECTED_ROUTABLE}
    for playbook in playbooks.routable_playbooks():
        prop = schema["properties"][playbooks._field_name(playbook.id)]
        assert prop["type"] == "boolean"
        assert playbook.description in prop["description"]


async def test_select_playbooks_returns_ids_answered_true():
    with patch.object(playbooks.Agent, "run", AsyncMock(return_value=_run_result(_selection("pain-and-injury")))):
        assert await playbooks.select_playbooks("my achilles hurts") == ["pain-and-injury"]


async def test_select_playbooks_returns_empty_when_nothing_matches():
    with patch.object(playbooks.Agent, "run", AsyncMock(return_value=_run_result(_selection()))):
        assert await playbooks.select_playbooks("thanks!") == []


async def test_select_playbooks_keeps_the_most_confident_when_over_the_cap():
    output = _selection("training-week", "pain-and-injury", "recovery-and-readiness")
    confidence = {"training_week": 0.1, "pain_and_injury": 0.9, "recovery_and_readiness": 0.5}
    with patch.object(playbooks.Agent, "run", AsyncMock(return_value=_run_result(output, confidence))):
        picked = await playbooks.select_playbooks("knee hurts, marathon in 6 weeks")

    assert picked == ["pain-and-injury", "recovery-and-readiness"]


async def test_select_playbooks_fails_open():
    with patch.object(playbooks.Agent, "run", AsyncMock(side_effect=RuntimeError("jev down"))):
        assert await playbooks.select_playbooks("how do I taper?") == []


async def test_router_prompt_includes_only_recent_history():
    history = []
    for i in range(6):
        history.append(ModelRequest(parts=[UserPromptPart(content=f"user turn {i}")]))
        history.append(ModelResponse(parts=[TextPart(content=f"coach turn {i}")]))
    run = AsyncMock(return_value=_run_result(_selection()))

    with patch.object(playbooks.Agent, "run", run):
        await playbooks.select_playbooks("and for a 70.3?", history)

    assert run.await_args is not None
    prompt = run.await_args.args[0]
    assert "LATEST MESSAGE TO CLASSIFY: and for a 70.3?" in prompt
    assert "user turn 5" in prompt and "coach turn 4" in prompt
    assert "user turn 3" not in prompt


def test_render_instructions():
    assert playbooks.render_instructions([]) is None
    assert playbooks.render_instructions(["no-such-playbook"]) is None

    rendered = playbooks.render_instructions(["race-execution", "fueling-and-hydration"])
    assert rendered is not None
    assert "## Taper and race execution" in rendered
    assert "## Fueling, hydration and everyday nutrition" in rendered
    assert "never mention, quote or name this guidance" in rendered
