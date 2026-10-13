"""Coaching playbooks: short, opinionated guidance per topic, injected into the turns
that need it.

Content lives in `app/playbooks/*.md` (format and writing rules:
`app/playbooks/README.md`). `coaching-core.md` is always part of the system prompt
(`coaching_core()`). The rest are routed: before the coach agent runs,
`select_playbooks` asks Jev (the typed decision model, `get_jev_model()`) one yes/no
per playbook, and `render_instructions` turns the picks into per-run `instructions`
for that turn only. The base prompt stays small, and adding a topic is just adding a
file.

Replaces the old Qdrant-backed knowledge-base search: a few curated files don't need
embeddings or an ingestion step, and routing every turn (instead of hoping the model
calls a search tool) means relevant guidance is never skipped.

Fails open: any router error injects nothing, and chat runs on the base prompt alone.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, create_model
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse

from app.agents.models import get_jev_model
from app.services.scope import flatten_history_for_prompt

logger = logging.getLogger(__name__)

PLAYBOOK_DIR = Path(__file__).resolve().parent.parent / "playbooks"
CORE_ID = "coaching-core"
# Caps the context a turn can pull in. Guidance from more than two topics at once is
# rarely what a single message needs, and each playbook is ~1k tokens.
MAX_PLAYBOOKS_PER_TURN = 2
# Prior messages the router sees: enough for a short follow-up ("and for a 70.3?") to
# keep its topic, without paying for a long conversation on every turn.
_ROUTER_HISTORY_MESSAGES = 4

_ROUTER_INSTRUCTIONS = (
    "An endurance-sports coach (running, cycling, swimming, triathlon) is about to "
    "answer the LATEST MESSAGE below. For each coaching playbook, decide whether its "
    "guidance would directly help answer that message. Use the prior conversation only "
    "to understand short follow-ups. Most messages need one playbook; some need two; "
    "greetings, thanks and questions answered by a single number need none."
)


@dataclass(frozen=True)
class Playbook:
    id: str
    title: str
    description: str
    body: str


def _parse_playbook(path: Path) -> Playbook:
    """Read a playbook file: `---` frontmatter (key: value, indented continuation
    lines), then the markdown body. The `## Sources` section is for human review and
    is dropped."""
    text = path.read_text(encoding="utf-8")
    _, frontmatter, body = text.split("---\n", 2)

    meta: dict[str, str] = {}
    key = None
    for line in frontmatter.splitlines():
        if line[:1].isspace() and key is not None:
            meta[key] = f"{meta[key]} {line.strip()}"
        elif ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            meta[key] = value.split("#", 1)[0].strip()

    body = body.split("\n## Sources", 1)[0].strip()
    playbook = Playbook(id=meta["id"], title=meta["title"], description=meta["description"], body=body)
    if playbook.id != path.stem:
        raise ValueError(f"Playbook id {playbook.id!r} doesn't match its filename {path.name!r}")
    return playbook


@lru_cache
def load_playbooks() -> dict[str, Playbook]:
    """Every playbook (including the always-on core), keyed by id."""
    playbooks = [_parse_playbook(path) for path in sorted(PLAYBOOK_DIR.glob("*.md")) if path.name != "README.md"]
    return {playbook.id: playbook for playbook in playbooks}


def coaching_core() -> str:
    """The always-on coaching guidance appended to the coach's system prompt."""
    return load_playbooks()[CORE_ID].body


def routable_playbooks() -> list[Playbook]:
    return [playbook for playbook in load_playbooks().values() if playbook.id != CORE_ID]


def _field_name(playbook_id: str) -> str:
    return playbook_id.replace("-", "_")


@lru_cache
def _selection_type() -> type[BaseModel]:
    """Output type for the router: one yes/no per playbook, which Jev answers in one
    request. A field's description is the question Jev is asked about it."""
    fields = {
        _field_name(playbook.id): (
            bool,
            Field(description=f"Would the '{playbook.title}' playbook help? It covers: {playbook.description}"),
        )
        for playbook in routable_playbooks()
    }
    return create_model(  # pyright: ignore[reportCallIssue]
        "PlaybookSelection",
        __doc__="Which coaching playbooks are relevant to the latest message.",
        **fields,  # pyright: ignore[reportArgumentType]
    )


def _confidences(result_messages: list[ModelMessage]) -> dict[str, float]:
    """Per-field confidence Jev reports in `provider_details` (distance from the yes/no
    threshold, so among fields answered True, higher means a surer yes).

    Only used to rank when more than MAX_PLAYBOOKS_PER_TURN come back true, so a
    missing or differently-shaped value just falls back to file order.
    """
    responses = [m for m in result_messages if isinstance(m, ModelResponse)]
    details = (responses[-1].provider_details if responses else None) or {}
    raw = details.get("confidence") or {}
    return {name: value for name, value in raw.items() if isinstance(value, int | float)}


async def select_playbooks(message: str, message_history: list[ModelMessage] | None = None) -> list[str]:
    """Ids of the playbooks (at most MAX_PLAYBOOKS_PER_TURN) relevant to `message`.

    Fails open: returns [] on any router error, so chat carries on with the base
    prompt.
    """
    try:
        agent = Agent(model=get_jev_model(), output_type=_selection_type(), instructions=_ROUTER_INSTRUCTIONS)
        history = (message_history or [])[-_ROUTER_HISTORY_MESSAGES:]
        result = await agent.run(flatten_history_for_prompt(message, history))
    except Exception:
        logger.exception("Playbook router (Jev) failed; continuing without playbooks")
        return []

    picked = [playbook.id for playbook in routable_playbooks() if getattr(result.output, _field_name(playbook.id))]
    if len(picked) > MAX_PLAYBOOKS_PER_TURN:
        confidence = _confidences(result.all_messages())
        picked.sort(key=lambda pid: confidence.get(_field_name(pid), 0.0), reverse=True)
        picked = picked[:MAX_PLAYBOOKS_PER_TURN]
    logger.info("Playbooks for this turn: %s", picked or "none")
    return picked


def render_instructions(playbook_ids: list[str]) -> str | None:
    """Per-run instructions carrying the chosen playbooks, or None when there are none."""
    playbooks = load_playbooks()
    sections = [f"## {playbooks[pid].title}\n\n{playbooks[pid].body}" for pid in playbook_ids if pid in playbooks]
    if not sections:
        return None
    return (
        "Coaching guidance for this question. These are this coach's own positions: "
        "follow them over generic advice, and ask the 'Ask first' questions only when "
        "the answer depends on them. Speak as the coach giving your own advice: never "
        "mention, quote or name this guidance (don't call it a playbook or a plan you "
        "were given).\n\n" + "\n\n".join(sections)
    )
