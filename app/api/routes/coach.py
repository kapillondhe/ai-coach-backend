import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Literal, NamedTuple

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart

from app.agents.coach_agent import CoachReply, get_coach_agent
from app.core.auth import get_current_user_id
from app.core.rate_limit import enforce_chat_rate_limit
from app.services import chat_opener as chat_opener_service
from app.services import conversation as conversation_service
from app.services import safety as safety_service
from app.services import scope as scope_service
from app.services import titling as titling_service
from app.services.conversation import ConversationNotFoundError, MessageData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/coach", tags=["coach"])

_AGENT_UNAVAILABLE_DETAIL = "Coach agent is temporarily unavailable. Please try again."

# Request size caps (over-limit requests get a 422). Anonymous users send their own
# history, assistant turns included; that is accepted by design, only bounded.
_MAX_MESSAGE_CHARS = 4000
_MAX_HISTORY_TURNS = 50
_MAX_HISTORY_TURN_CHARS = 8000


class ChatOpenerResponse(BaseModel):
    text: str
    chips: list[str] = []


@router.get("/opener")
async def get_chat_opener(user_id: str | None = Depends(get_current_user_id)) -> ChatOpenerResponse:
    opener = await chat_opener_service.get_chat_opener(user_id)
    return ChatOpenerResponse(text=opener.text, chips=opener.chips)


class ChatHistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=_MAX_HISTORY_TURN_CHARS)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=_MAX_MESSAGE_CHARS)
    session_id: str | None = None
    history: list[ChatHistoryTurn] = Field(default=[], max_length=_MAX_HISTORY_TURNS)
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str | None = None
    suggestions: list[str] = []


class _TurnContext(NamedTuple):
    message_history: list[ModelMessage]
    conversation_id: str | None
    is_new_conversation: bool


def _turns_to_message_history(turns: list[tuple[str, str]]) -> list[ModelMessage]:
    messages: list[ModelMessage] = []
    for role, content in turns:
        if role == "user":
            messages.append(ModelRequest(parts=[UserPromptPart(content=content)]))
        elif role == "assistant":
            messages.append(ModelResponse(parts=[TextPart(content=content)]))
    return messages


async def _resolve_context(request: ChatRequest, user_id: str | None) -> _TurnContext:
    if user_id is None:
        turns = [(turn.role, turn.content) for turn in request.history]
        return _TurnContext(_turns_to_message_history(turns), None, False)

    if request.conversation_id:
        # The previous turn's save may still be in flight (it runs after `done` is sent);
        # wait for it so this turn's history includes it. Only works within one process:
        # see `_pending_writes`.
        pending = _pending_writes.get(request.conversation_id)
        if pending is not None:
            await asyncio.wait([pending])
        try:
            data = await conversation_service.get_conversation(request.conversation_id, user_id)
        except ConversationNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Conversation not found") from exc
        history = _turns_to_message_history([(m.role, m.content) for m in data.messages])
        return _TurnContext(history, data.id, False)

    return _TurnContext([], await conversation_service.create_conversation(user_id), True)


async def _start_context_load(
    request: ChatRequest, user_id: str | None = Depends(get_current_user_id)
) -> asyncio.Task[_TurnContext]:
    """Kick off `_resolve_context` as a task so it overlaps with `get_coach_agent`'s setup.

    Must be declared before the `agent` dependency in a route's signature: FastAPI
    resolves dependencies in order, so the DB load is already running while the
    agent's toolsets/memories are built. The route awaits the task. Route-level
    `dependencies=[...]` (the rate limit) run before it, so a rejected request never
    starts the load.
    """
    task = asyncio.create_task(_resolve_context(request, user_id))
    # Mark any exception retrieved if the route never awaits it (e.g. agent setup raised).
    task.add_done_callback(lambda t: t.cancelled() or t.exception())
    return task


async def _persist_turn(user_id: str | None, conversation_id: str | None, user_message: str, reply: str) -> None:
    if user_id is None or conversation_id is None:
        return
    await conversation_service.append_messages(
        conversation_id,
        user_id,
        [MessageData(role="user", content=user_message), MessageData(role="assistant", content=reply)],
    )


# In-flight streamed-turn saves by conversation_id: holds a strong reference so the
# task isn't garbage-collected, and lets the next turn wait for it (`_resolve_context`).
# Assumes a single worker process: this dict is per-process, so with multiple
# workers/replicas the next turn can land elsewhere and load history before this save
# commits. If the app ever scales out, move the save back before the `done` event.
_pending_writes: dict[str, asyncio.Task[None]] = {}


def _persist_turn_in_background(
    user_id: str | None, conversation_id: str | None, user_message: str, reply: str
) -> asyncio.Task[None] | None:
    """Start saving a turn without blocking the stream's `done` event; failures are logged."""
    if user_id is None or conversation_id is None:
        return None
    task = asyncio.create_task(_persist_turn(user_id, conversation_id, user_message, reply))
    _pending_writes[conversation_id] = task

    def _on_done(t: asyncio.Task[None]) -> None:
        if _pending_writes.get(conversation_id) is t:
            del _pending_writes[conversation_id]
        if not t.cancelled() and t.exception() is not None:
            logger.error("Failed to persist chat turn for conversation %s", conversation_id, exc_info=t.exception())

    task.add_done_callback(_on_done)
    return task


async def _short_circuit_reply(message: str, message_history: list[ModelMessage]) -> str | None:
    """A canned reply for messages the agent shouldn't handle (emergency, off-topic), else None."""
    if safety_service.contains_emergency_red_flag(message):
        logger.warning("Emergency red flag detected in chat message; short-circuiting the agent")
        return safety_service.EMERGENCY_RESPONSE
    if await scope_service.is_off_topic(message, message_history=message_history):
        logger.info("Off-topic message detected; short-circuiting the agent")
        return scope_service.OFF_TOPIC_RESPONSE
    return None


def _schedule_titling(
    background_tasks: BackgroundTasks,
    user_id: str | None,
    conversation_id: str | None,
    user_message: str,
    is_new_conversation: bool,
) -> None:
    if is_new_conversation and user_id is not None and conversation_id is not None:
        background_tasks.add_task(titling_service.generate_and_set_title, conversation_id, user_id, user_message)


def _sse(data: dict, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data)}\n\n"


def _tag_session(request: ChatRequest) -> None:
    if request.session_id:
        trace.get_current_span().set_attribute(SpanAttributes.SESSION_ID, request.session_id)


@router.post("/chat", dependencies=[Depends(enforce_chat_rate_limit)])
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    context_load: asyncio.Task[_TurnContext] = Depends(_start_context_load),
    agent: Agent[None, CoachReply] = Depends(get_coach_agent),
    user_id: str | None = Depends(get_current_user_id),
) -> ChatResponse:
    _tag_session(request)
    message_history, conversation_id, is_new_conversation = await context_load

    reply = await _short_circuit_reply(request.message, message_history)
    suggestions: list[str] = []
    if reply is None:
        try:
            result = await agent.run(request.message, message_history=message_history)
        except Exception as exc:
            logger.exception("Coach agent failed to produce a reply")
            raise HTTPException(status_code=502, detail=_AGENT_UNAVAILABLE_DETAIL) from exc
        reply = result.output.reply
        suggestions = result.output.suggestions

    await _persist_turn(user_id, conversation_id, request.message, reply)
    _schedule_titling(background_tasks, user_id, conversation_id, request.message, is_new_conversation)
    return ChatResponse(reply=reply, conversation_id=conversation_id, suggestions=suggestions)


@router.post("/chat/stream", dependencies=[Depends(enforce_chat_rate_limit)])
async def chat_stream(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    context_load: asyncio.Task[_TurnContext] = Depends(_start_context_load),
    agent: Agent[None, CoachReply] = Depends(get_coach_agent),
    user_id: str | None = Depends(get_current_user_id),
) -> StreamingResponse:
    _tag_session(request)
    message_history, conversation_id, is_new_conversation = await context_load

    async def event_generator() -> AsyncIterator[str]:
        write: asyncio.Task[None] | None = None
        try:
            if conversation_id:
                yield _sse({"conversation_id": conversation_id}, "conversation")

            reply = await _short_circuit_reply(request.message, message_history)
            suggestions: list[str] = []
            if reply is not None:
                yield _sse({"delta": reply})
            else:
                sent_so_far = ""
                async with agent.run_stream(request.message, message_history=message_history) as result:
                    async for partial in result.stream_output(debounce_by=0.1):
                        reply_so_far = partial.reply or ""
                        if len(reply_so_far) > len(sent_so_far):
                            yield _sse({"delta": reply_so_far[len(sent_so_far) :]})
                            sent_so_far = reply_so_far
                    final_output = await result.get_output()
                    suggestions = final_output.suggestions
                    if len(final_output.reply) > len(sent_so_far):
                        yield _sse({"delta": final_output.reply[len(sent_so_far) :]})
                        sent_so_far = final_output.reply
                reply = sent_so_far

            # Save concurrently with sending the tail events instead of before them, so
            # the client's "loading" state ends ~1s sooner (the DB is a slow round trip).
            write = _persist_turn_in_background(user_id, conversation_id, request.message, reply)
            _schedule_titling(background_tasks, user_id, conversation_id, request.message, is_new_conversation)
            if suggestions:
                yield _sse({"suggestions": suggestions}, "suggestions")
            yield _sse({}, "done")
        except Exception:
            logger.exception("Coach agent failed to stream a reply")
            yield _sse({"detail": _AGENT_UNAVAILABLE_DETAIL}, "error")
        finally:
            # Keep the request open until the save lands (asyncio.wait neither raises the
            # task's error nor cancels it if this generator is cancelled on disconnect).
            if write is not None:
                await asyncio.wait([write])

    return StreamingResponse(event_generator(), media_type="text/event-stream")
