"""Memory-save guardrail: classify whether a candidate fact is actually worth
persisting to long-term user memory, as a check in front of the main coach
model's own judgment call.

This exists for the same reason as `app.services.scope`'s off-topic guardrail:
relying solely on the main coach model's judgment (via `coach_agent`'s
`_MEMORY_TOOL_DESCRIPTION`, "only call this for things worth recalling later —
not every message") is not a hard guarantee — a model can still call the
`remember` tool for a trivial, vague, or already-generic detail. This adds a
layer in front of the persist call that double-checks the decision instead of
trusting the main model's instruction-following alone.

Classification runs on Jev (TypeSafe's "System One" decision model) via
`app.agents.coach_agent.get_jev_model()` — the same model/transport as the
scope guardrail, for the same latency/cost reasons (see
`app.services.scope`'s module docstring).

Fails open (treats the fact as worth remembering) on any classifier error, so
a transient model/API issue never silently drops a fact the main model already
decided was worth saving.
"""

import logging
from typing import Annotated

from pydantic_ai import Agent, BoolCriteria

logger = logging.getLogger(__name__)

_MEMORY_INSTRUCTIONS = (
    "An endurance-sports coaching chat assistant wants to save the FACT below "
    "to long-term memory about a user, so it can be recalled in future "
    "conversations. Is this actually worth persisting long-term?"
)
_WORTH_REMEMBERING_CRITERIA = BoolCriteria(
    true=(
        "A durable, specific, personally-relevant detail about the user (an "
        "injury, a goal race, a dietary preference or restriction, a "
        "recurring training constraint, a meaningful preference) that would "
        "still be useful to know in a future conversation."
    ),
    false=(
        "Trivial, vague, one-off, already-generic, or scoped only to the "
        "current message — not something that needs to persist beyond this "
        "conversation."
    ),
)


async def is_worth_remembering(fact: str) -> bool:
    """True if `fact` is durable/specific enough to persist to long-term memory.

    Fails open (returns True) on any classifier error, so a transient
    model/API issue never silently drops a fact the main model already
    decided was worth saving.
    """
    try:
        # Imported lazily (not at module level) to avoid a circular import:
        # app.agents.coach_agent imports this module to call is_worth_remembering
        # from its `remember` tool.
        from app.agents.coach_agent import get_jev_model

        agent = Agent(
            model=get_jev_model(),
            output_type=Annotated[bool, _WORTH_REMEMBERING_CRITERIA],
            instructions=_MEMORY_INSTRUCTIONS,
        )
        result = await agent.run(f"FACT: {fact}")
        return result.output
    except Exception:
        logger.exception("Memory-save classifier (Jev) failed; defaulting to remembering the fact")
        return True
