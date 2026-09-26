
import logging

from pydantic_ai import Agent

from app.agents.coach_agent import get_utility_model
from app.services import conversation as conversation_service

logger = logging.getLogger(__name__)

_TITLE_SYSTEM_PROMPT = (
    "You generate short titles for chat conversations, like ChatGPT does. Given the user's "
    "first message to a fitness coaching assistant, reply with ONLY a concise title "
    "(3-6 words, no quotes, no punctuation at the end, no leading label like 'Title:'). "
    "Capture the topic, not a greeting."
)

_MAX_TITLE_LENGTH = 80


def _clean_title(raw: str) -> str:
    title = raw.strip().strip("\"'").splitlines()[0].strip()
    if len(title) > _MAX_TITLE_LENGTH:
        title = title[:_MAX_TITLE_LENGTH].rstrip()
    return title


async def generate_and_set_title(conversation_id: str, user_id: str, first_message: str) -> None:
    try:
        agent = Agent(model=get_utility_model(), system_prompt=_TITLE_SYSTEM_PROMPT)
        result = await agent.run(first_message)
        title = _clean_title(result.output)
        if not title:
            return
        await conversation_service.set_title(conversation_id, user_id, title)
    except Exception:
        logger.exception("Failed to generate title for conversation %s", conversation_id)
