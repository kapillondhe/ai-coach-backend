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

Unlike the emergency check, off-topic phrasing is too open-ended for a fixed
regex set, so this uses a second, cheap LLM call (the tier-2 utility model
already used for titling) as a classifier instead. It fails open (treats the
message as on-topic) on any classifier error, so a transient model/API issue
never blocks a legitimate training question from reaching the main agent.
"""

import logging

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart

from app.agents.coach_agent import get_utility_model

logger = logging.getLogger(__name__)

OFF_TOPIC_RESPONSE = (
    "That's outside what I help with — I'm here for endurance-sports training, "
    "nutrition, recovery, gear, and racing. Want help with any of that instead?"
)

_SCOPE_CLASSIFIER_PROMPT = (
    "Classify whether the LATEST MESSAGE below is ON-TOPIC for an "
    "endurance-sports (running, cycling, swimming, triathlon) training coach: "
    "training, nutrition, recovery, injury, gear, racing, a greeting/meta "
    "question about the coach itself, asking the coach to recall something "
    "said earlier in the prior conversation shown below, or any other short "
    "follow-up that only makes sense relative to that prior conversation all "
    "count as on-topic. Anything else (movies, actors, celebrities, trivia, "
    "coding help, general life advice, politics, hypotheticals unrelated to "
    "sport) is off-topic, even if raised mid-conversation. Reply with exactly "
    "`true` if ON-TOPIC, `false` if OFF-TOPIC. If the latest message mixes an "
    "off-topic question with on-topic content, classify as off-topic only if "
    "the primary ask is the off-topic part."
)


def _flatten_history_for_prompt(message: str, message_history: list[ModelMessage] | None) -> str:
    """Render prior turns as plain text for the classifier prompt.

    Deliberately does NOT pass `message_history` as real pydantic-ai history to
    `agent.run`: doing so was tried and regressed — with real history attached, the
    model (observed on z-ai/glm-5.3-flash) got confused about the generic
    `final_result` boolean output tool and started defaulting to `true` without
    actually reasoning about the classification (verified by inspecting
    `result.all_messages()`: a `ThinkingPart` second-guessing what the boolean tool
    even means, for a message that plainly should classify `False`). Flattening the
    history into the system prompt as inert text sidesteps that confusion entirely
    while still giving the classifier the context it needs for follow-ups like
    "what's my name?".
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
            model=get_utility_model(),
            output_type=bool,
            system_prompt=_SCOPE_CLASSIFIER_PROMPT,
        )
        prompt = _flatten_history_for_prompt(message, message_history)
        result = await agent.run(prompt)
        on_topic = result.output
        return not on_topic
    except Exception:
        logger.exception("Scope classifier failed; treating message as on-topic")
        return False
