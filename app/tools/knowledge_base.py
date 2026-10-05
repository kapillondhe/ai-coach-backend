import asyncio
import logging
from typing import Literal

from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)

KnowledgeDomain = Literal["nutrition", "exercises", "training", "physiotherapy"]

# Below this cosine-similarity score a match is considered noise rather than a real
# answer, so the tool reports "no relevant knowledge found" instead of returning it.
# Tuned for all-minilm-l6-v2, which scores lower than the previous bge model: off-topic
# queries top out around 0.2, while relevant passages (incl. medical red flags) land
# from ~0.33 up. Re-tune if EMBEDDING_MODEL changes.
_SCORE_THRESHOLD = 0.3
# Caps how many passages the model can pull into its own context in one call.
_MAX_TOP_K = 10


async def search_knowledge_base(
    query: str,
    domain: KnowledgeDomain | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Search the curated sports-science and exercise knowledge base for grounding.

    Use this whenever giving nutrition, exercise, training-plan, or injury/rehab advice
    that should be backed by a source rather than general knowledge — for example:
    specific macro or protein targets, race-day fueling and carb loading, safe exercise
    substitutions for an injury (e.g. "what can I do instead of squats with a bad
    knee"), movement-form guidance, marathon/endurance training-plan structure and
    pacing, sports anatomy, or how a tendon/joint injury is classified, rehabbed, and
    when it needs referral to a medical professional. Prefer this over answering from
    memory when the question is the kind a sports-science reference, physical
    therapist, or running coach would be consulted for.

    Args:
        query: The question or topic to search for, in plain language (e.g.
            "safe shoulder-friendly alternative to overhead press").
        domain: Restrict the search to one knowledge area, or leave unset to search
            all of them:
            - "nutrition": macros, protein/calorie targets, meal timing, race-day
              fueling and carb loading.
            - "exercises": exercise form, movement substitutions, injury workarounds.
            - "training": training-plan structure, weekly mileage/periodization,
              tapering, race-day pacing.
            - "physiotherapy": sports anatomy, injury classification and tissue
              healing, rehab and return-to-sport principles, common tendon/joint
              conditions, and medical red flags that warrant referral rather than
              self-management.
        top_k: Maximum number of matching passages to return, from 1 to 10.
            Defaults to 5.

    Returns:
        A list of matched passages, each a dict with `text` (the passage content),
        `source` (the knowledge-base file it came from, for citing), `domain`, and
        `score` (similarity, higher is more relevant). Returns an empty list when
        nothing in the knowledge base is relevant enough to the query, or if the
        knowledge base is temporarily unavailable — in either case, fall back to
        general knowledge and don't claim a source.
    """
    top_k = max(1, min(top_k, _MAX_TOP_K))
    try:
        store = get_vector_store()
        return await asyncio.to_thread(store.search, query, domain, top_k, _SCORE_THRESHOLD)
    except Exception as exc:
        # One line, not a traceback: an unreachable/empty Qdrant is an expected,
        # recoverable state (chat continues ungrounded), not a crash. str() over repr():
        # qdrant's UnexpectedResponse reprs as a bare "UnexpectedResponse()", hiding the
        # status code that tells a bad API key (403) from a missing collection (404).
        detail = " ".join(str(exc).split())[:300]
        logger.warning("Knowledge base search failed; continuing without grounding: %s: %s", type(exc).__name__, detail)
        return []
