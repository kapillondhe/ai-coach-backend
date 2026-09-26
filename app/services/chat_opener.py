
from dataclasses import dataclass, field

from sqlalchemy import select

from app.core.db import get_session
from app.core.models import SyncedActivity, UserIntegration

_ANONYMOUS_OPENER = "What are you training for?"
_ANONYMOUS_CHIPS = [
    "I'm training for my first 70.3",
    "How much protein do I need?",
    "Explain zone 2 training",
]

_SIGNED_IN_OPENER = "Good to see you. What's on your mind for training this week?"
_SIGNED_IN_CHIPS = [
    "Log how today's run felt",
    "What should I eat before a long ride?",
    "How should I structure a long run?",
]

_CONNECTED_CHIPS = [
    "How's my training load this week?",
    "Log how today's run felt",
    "What should I eat before a long ride?",
]

_DISCIPLINE_WORDS = {"run": "run", "bike": "ride", "swim": "swim"}


@dataclass
class ChatOpener:
    text: str
    chips: list[str] = field(default_factory=list)


async def _is_coros_connected(user_id: str) -> bool:
    async with get_session() as db:
        row = (
            await db.execute(
                select(UserIntegration.provider).where(
                    UserIntegration.user_id == user_id, UserIntegration.provider == "coros"
                )
            )
        ).first()
    return row is not None


async def _latest_activity(user_id: str) -> SyncedActivity | None:
    async with get_session() as db:
        result = await db.execute(
            select(SyncedActivity)
            .where(SyncedActivity.user_id == user_id)
            .order_by(SyncedActivity.started_at.desc().nullslast())
            .limit(1)
        )
    return result.scalars().first()


def _describe_activity(activity: SyncedActivity) -> str:
    discipline_word = _DISCIPLINE_WORDS.get(activity.discipline, "session")
    when = activity.started_at.strftime("%A") if activity.started_at else "recently"

    details = []
    if activity.distance_km:
        details.append(f"{activity.distance_km:.1f}km")
    if activity.duration_seconds:
        details.append(f"{round(activity.duration_seconds / 60)} min")
    detail_str = f" — {', '.join(details)}" if details else ""

    return f"Nice {discipline_word} on {when}{detail_str}. How are you feeling for what's next?"


async def get_chat_opener(user_id: str | None) -> ChatOpener:
    if user_id is None:
        return ChatOpener(text=_ANONYMOUS_OPENER, chips=list(_ANONYMOUS_CHIPS))

    if not await _is_coros_connected(user_id):
        return ChatOpener(text=_SIGNED_IN_OPENER, chips=list(_SIGNED_IN_CHIPS))

    activity = await _latest_activity(user_id)
    if activity is None:
        return ChatOpener(text=_SIGNED_IN_OPENER, chips=list(_SIGNED_IN_CHIPS))

    return ChatOpener(text=_describe_activity(activity), chips=list(_CONNECTED_CHIPS))
