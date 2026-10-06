"""Scope guardrail: classify whether a message is on-topic for the coach, and
short-circuit off-topic messages to a canned redirect before the main coach
agent ever runs.

Why: `coach_agent.SYSTEM_PROMPT`'s scope rule alone isn't reliable; models still
partly answer casual off-topic asides after a few friendly turns. Like
`app.services.safety`'s emergency pre-check, this layer can't be talked out of
declining.

How: a pydantic-ai `Agent` on Jev (`app.agents.models.get_jev_model()`), a typed
yes/no decision model that was ~5x faster and 3-4x cheaper per call than the
tier-2 utility model it replaced. It runs before every message, so latency matters.

OPEN ISSUE: after switching to `BoolCriteria`/`Annotated[bool, ...]` output, Jev
scored 12/14 vs the legacy classifier's 14/14 on evals/cases.py's labeled set;
accuracy still needs re-verification before it's trusted at face value.

Fails open (treats the message as on-topic) on any classifier error.
"""

import logging
from typing import Annotated

from pydantic_ai import Agent, BoolCriteria
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart

from app.agents.models import get_jev_model

logger = logging.getLogger(__name__)

OFF_TOPIC_RESPONSE = (
    "That's outside what I help with — I'm here for endurance-sports training, "
    "nutrition, recovery, gear, and racing. Want help with any of that instead?"
)

_SCOPE_INSTRUCTIONS = (
    "Is the LATEST MESSAGE below on-topic for an endurance-sports (running, "
    "cycling, swimming, triathlon) training coach: training, nutrition, "
    "recovery, injury, gear, racing, a greeting/meta question about the coach "
    "itself, asking the coach to recall something said earlier in the prior "
    "conversation shown below, or any other short follow-up that only makes "
    "sense relative to that prior conversation? Anything else (movies, actors, "
    "celebrities, trivia, coding help, general life advice, politics, "
    "hypotheticals unrelated to sport) is off-topic, even if raised "
    "mid-conversation. If the latest message mixes an off-topic question with "
    "on-topic content, answer false only if the primary ask is the off-topic "
    "part."
)
_ON_TOPIC_CRITERIA = BoolCriteria(
    true=(
        "Training, nutrition, recovery, injury, gear, racing, a greeting/meta "
        "question about the coach, or a follow-up that only makes sense "
        "relative to the prior conversation shown."
    ),
    false=(
        "Movies, actors, celebrities, trivia, coding help, general life advice, "
        "politics, or any hypothetical unrelated to sport — even if the "
        "primary ask is off-topic while some on-topic content is also present."
    ),
)


def _flatten_history_for_prompt(message: str, message_history: list[ModelMessage] | None) -> str:
    """Render prior turns as plain text for the classifier prompt.

    Same reasoning as the pre-Jev approach: flattening history into inert text
    (rather than passing it as real pydantic-ai `message_history`) keeps the
    classifier focused purely on the yes/no judgment, while still giving it
    the context it needs for follow-ups like "what's my name?".
    """
    if not message_history:
        return f"LATEST MESSAGE TO CLASSIFY: {message}"

    lines = ["Prior conversation (context only, do not classify this part):"]
    for turn in message_history:
        if isinstance(turn, ModelRequest):
            for part in turn.parts:
                if isinstance(part, UserPromptPart):
                    lines.append(f"user: {part.content}")
        elif isinstance(turn, ModelResponse):
            for part in turn.parts:
                if isinstance(part, TextPart):
                    lines.append(f"assistant: {part.content}")
    lines.append("")
    lines.append(f"LATEST MESSAGE TO CLASSIFY: {message}")
    return "\n".join(lines)


async def is_off_topic(message: str, message_history: list[ModelMessage] | None = None) -> bool:
    """True if `message`'s primary ask is unrelated to endurance-sports coaching.

    `message_history` (same shape the main coach agent is called with) gives the
    classifier the prior turns as context, flattened into the prompt as plain text
    (see `_flatten_history_for_prompt`), so a short follow-up that only makes sense
    relative to an on-topic conversation (e.g. "what's my name?" right after
    introducing it) isn't misclassified as off-topic in isolation.

    Fails open (returns False, i.e. "treat as on-topic") on any classifier
    error, so a transient model/API issue never blocks a legitimate training
    question from reaching the main agent.
    """
    try:
        agent = Agent(
            model=get_jev_model(),
            output_type=Annotated[bool, _ON_TOPIC_CRITERIA],
            instructions=_SCOPE_INSTRUCTIONS,
        )
        prompt = _flatten_history_for_prompt(message, message_history)
        result = await agent.run(prompt)
        on_topic = result.output
        return not on_topic
    except Exception:
        logger.exception("Scope classifier (Jev) failed; treating message as on-topic")
        return False
