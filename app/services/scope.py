"""Scope guardrail: classify whether a message is on-topic for the coach, and
short-circuit off-topic messages to a canned redirect before the main coach
agent ever runs.

This exists because `coach_agent.SYSTEM_PROMPT`'s "Scope" instruction is not
reliable enough on its own: even with an explicit "give zero opinion on the
off-topic subject" rule and a worked example, the configured model (observed
on z-ai/glm-5.3-flash) was still caught answering part of an off-topic
question ("I'd give the edge to Denzel...") before redirecting — especially
when the question is phrased casually ("just curious", "if you had to pick")
after a few friendly on-topic turns. Same failure mode, and same fix shape, as
`app.services.safety`'s emergency-regex pre-check: prompt instructions alone
aren't a hard guarantee, so add a layer in front of the main model that can't
be talked out of declining.

Classification runs on Jev (TypeSafe's "System One" decision model) via
`app.agents.coach_agent.get_jev_model()` — a real pydantic-ai `Agent` with
`output_type=bool`, same as the pre-Jev approach, but on a `SystemOneModel`
instead of a chat-completion model: Jev answers a typed yes/no question with
a probability instead of generating and parsing free text. Routed through
OpenRouter (same OPENROUTER_API_KEY as every other model here), so no
separate TypeSafe account is needed. `pydantic-ai>=2.45` is required for
`pydantic_ai.models.system_one`/`pydantic_ai.providers.system_one` to exist.

Chosen over the previous tier-2-utility-model approach for latency and cost:
an early side-by-side on evals/cases.py's labeled cases showed roughly 5x
lower latency and 3-4x lower cost per call — the generative call's biggest
cost here was latency, since it runs before the user sees anything on every
single message. NOTE: a later rerun of that same comparison, after switching
to `BoolCriteria`/`Annotated[bool, ...]` for the output type, found Jev
scoring 12/14 against the legacy approach's 14/14 on the same labeled set —
accuracy parity has NOT been re-confirmed since that output-type change and
should be re-verified (build a labeled eval script again, or extend
evals/cases.py) before trusting this classifier's accuracy at face value.

Fails open (treats the message as on-topic) on any classifier error, so a
transient model/API issue never blocks a legitimate training question from
reaching the main agent.
"""

import logging
from typing import Annotated

from pydantic_ai import Agent, BoolCriteria
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart

from app.agents.coach_agent import get_jev_model

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
