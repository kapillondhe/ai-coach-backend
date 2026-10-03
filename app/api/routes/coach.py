import asyncio
import json
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart

from app.agents.coach_agent import CoachReply, get_coach_agent
from app.core.auth import get_current_user_id
from app.services import chat_opener as chat_opener_service
from app.services import conversation as conversation_service
from app.services import safety as safety_service
from app.services import titling as titling_service
from app.services.conversation import ConversationNotFoundError, MessageData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/coach", tags=["coach"])

_AGENT_UNAVAILABLE_DETAIL = "Coach agent is temporarily unavailable. Please try again."


class ChatOpenerResponse(BaseModel):
    text: str
    chips: list[str] = []


@router.get("/opener", response_model=ChatOpenerResponse)
async def get_chat_opener(user_id: str | None = Depends(get_current_user_id)) -> ChatOpenerResponse:
    opener = await chat_opener_service.get_chat_opener(user_id)
    return ChatOpenerResponse(text=opener.text, chips=opener.chips)


class ChatHistoryTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    history: list[ChatHistoryTurn] = []
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str | None = None
    suggestions: list[str] = []


def _turns_to_message_history(turns: list[tuple[str, str]]) -> list[ModelMessage]:
    messages: list[ModelMessage] = []
    for role, content in turns:
        if role == "user":
            messages.append(ModelRequest(parts=[UserPromptPart(content=content)]))
        elif role == "assistant":
            messages.append(ModelResponse(parts=[TextPart(content=content)]))
    return messages


def _to_message_history(history: list[ChatHistoryTurn]) -> list[ModelMessage]:
    return _turns_to_message_history([(turn.role, turn.content) for turn in history])


async def _resolve_context(request: ChatRequest, user_id: str | None) -> tuple[list[ModelMessage], str | None, bool]:
    """Returns (message_history, conversation_id, is_new_conversation)."""
    if user_id is None:
        return _to_message_history(request.history), None, False

    if request.conversation_id:
        # The previous turn's save may still be in flight (it runs after `done` is sent);
        # wait for it so this turn's history includes it.
        pending = _pending_writes.get(request.conversation_id)
        if pending is not None:
            await asyncio.wait([pending])
        try:
            data = await conversation_service.get_conversation(request.conversation_id, user_id)
        except ConversationNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Conversation not found") from exc
        history = _turns_to_message_history([(m.role, m.content) for m in data.messages])
        return history, data.id, False

    new_conversation_id = await conversation_service.create_conversation(user_id)
    return [], new_conversation_id, True


async def _start_context_load(
    request: ChatRequest, user_id: str | None = Depends(get_current_user_id)
) -> asyncio.Task[tuple[list[ModelMessage], str | None, bool]]:
    """Kick off `_resolve_context` as a task so it overlaps with `get_coach_agent`'s setup.

    Must be declared before the `agent` dependency in a route's signature: FastAPI
    resolves dependencies in order, so the DB load is already running while the
    agent's toolsets/memories are built. The route awaits the task.
    """
    task = asyncio.create_task(_resolve_context(request, user_id))
    # Mark any exception retrieved if the route never awaits it (e.g. agent setup raised).
    task.add_done_callback(lambda t: t.cancelled() or t.exception())
    return task


async def _persist_turn(
    user_id: str | None, conversation_id: str | None, user_message: str, reply: str
) -> None:
    if user_id is None or conversation_id is None:
        return
    await conversation_service.append_messages(
        conversation_id,
        user_id,
        [MessageData(role="user", content=user_message), MessageData(role="assistant", content=reply)],
    )


# In-flight streamed-turn saves by conversation_id: holds a strong reference so the
# task isn't garbage-collected, and lets the next turn wait for it (`_resolve_context`).
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
            logger.error(
                "Failed to persist chat turn for conversation %s", conversation_id, exc_info=t.exception()
            )

    task.add_done_callback(_on_done)
    return task


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    context_load: asyncio.Task = Depends(_start_context_load),
    agent: Agent[None, CoachReply] = Depends(get_coach_agent),
    user_id: str | None = Depends(get_current_user_id),
) -> ChatResponse:
    if request.session_id:
        trace.get_current_span().set_attribute(SpanAttributes.SESSION_ID, request.session_id)

    message_history, conversation_id, is_new_conversation = await context_load

    if safety_service.contains_emergency_red_flag(request.message):
        logger.warning("Emergency red flag detected in chat message; short-circuiting the agent")
        reply = safety_service.EMERGENCY_RESPONSE
        await _persist_turn(user_id, conversation_id, request.message, reply)
        if is_new_conversation and user_id is not None and conversation_id is not None:
            background_tasks.add_task(
                titling_service.generate_and_set_title, conversation_id, user_id, request.message
            )
        return ChatResponse(reply=reply, conversation_id=conversation_id, suggestions=[])

    try:
        result = await agent.run(request.message, message_history=message_history)
    except Exception as exc:
        logger.exception("Coach agent failed to produce a reply")
        raise HTTPException(status_code=502, detail=_AGENT_UNAVAILABLE_DETAIL) from exc

    reply = result.output.reply
    suggestions = result.output.suggestions

    await _persist_turn(user_id, conversation_id, request.message, reply)
    if is_new_conversation and user_id is not None and conversation_id is not None:
        background_tasks.add_task(
            titling_service.generate_and_set_title, conversation_id, user_id, request.message
        )
    return ChatResponse(reply=reply, conversation_id=conversation_id, suggestions=suggestions)


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    context_load: asyncio.Task = Depends(_start_context_load),
    agent: Agent[None, CoachReply] = Depends(get_coach_agent),
    user_id: str | None = Depends(get_current_user_id),
) -> StreamingResponse:
    if request.session_id:
        trace.get_current_span().set_attribute(SpanAttributes.SESSION_ID, request.session_id)

    message_history, conversation_id, is_new_conversation = await context_load

    async def event_generator():
        sent_so_far = ""
        suggestions: list[str] = []
        write: asyncio.Task[None] | None = None
        try:
            if conversation_id:
                yield f"event: conversation\ndata: {json.dumps({'conversation_id': conversation_id})}\n\n"

            if safety_service.contains_emergency_red_flag(request.message):
                logger.warning(
                    "Emergency red flag detected in chat message; short-circuiting the agent"
                )
                reply = safety_service.EMERGENCY_RESPONSE
                yield f"data: {json.dumps({'delta': reply})}\n\n"
                write = _persist_turn_in_background(user_id, conversation_id, request.message, reply)
                if is_new_conversation and user_id is not None and conversation_id is not None:
                    background_tasks.add_task(
                        titling_service.generate_and_set_title, conversation_id, user_id, request.message
                    )
                yield "event: done\ndata: {}\n\n"
            else:
                async with agent.run_stream(request.message, message_history=message_history) as result:
                    async for partial in result.stream_output(debounce_by=0.1):
                        reply_so_far = partial.reply if partial.reply else ""
                        if len(reply_so_far) > len(sent_so_far):
                            delta = reply_so_far[len(sent_so_far):]
                            sent_so_far = reply_so_far
                            yield f"data: {json.dumps({'delta': delta})}\n\n"
                    final_output = await result.get_output()
                    suggestions = final_output.suggestions
                    if len(final_output.reply) > len(sent_so_far):
                        delta = final_output.reply[len(sent_so_far):]
                        sent_so_far = final_output.reply
                        yield f"data: {json.dumps({'delta': delta})}\n\n"
                # Save concurrently with sending the tail events instead of before them, so
                # the client's "loading" state ends ~1s sooner (the DB is a slow round trip).
                write = _persist_turn_in_background(user_id, conversation_id, request.message, sent_so_far)
                if is_new_conversation and user_id is not None and conversation_id is not None:
                    background_tasks.add_task(
                        titling_service.generate_and_set_title, conversation_id, user_id, request.message
                    )
                if suggestions:
                    yield f"event: suggestions\ndata: {json.dumps({'suggestions': suggestions})}\n\n"
                yield "event: done\ndata: {}\n\n"
        except Exception:
            logger.exception("Coach agent failed to stream a reply")
            yield f"event: error\ndata: {json.dumps({'detail': _AGENT_UNAVAILABLE_DETAIL})}\n\n"
        finally:
            # Keep the request open until the save lands (asyncio.wait neither raises the
            # task's error nor cancels it if this generator is cancelled on disconnect).
            if write is not None:
                await asyncio.wait([write])

    return StreamingResponse(event_generator(), media_type="text/event-stream")
