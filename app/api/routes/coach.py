import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart

from app.agents.coach_agent import get_coach_agent
from app.core.auth import get_current_user_id
from app.services import conversation as conversation_service
from app.services.conversation import ConversationNotFoundError, MessageData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/coach", tags=["coach"])

_AGENT_UNAVAILABLE_DETAIL = "Coach agent is temporarily unavailable. Please try again."


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


async def _resolve_context(request: ChatRequest, user_id: str | None) -> tuple[list[ModelMessage], str | None]:
    if user_id is None:
        return _to_message_history(request.history), None

    if request.conversation_id:
        try:
            data = await conversation_service.get_conversation(request.conversation_id, user_id)
        except ConversationNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Conversation not found") from exc
        history = _turns_to_message_history([(m.role, m.content) for m in data.messages])
        return history, data.id

    new_conversation_id = await conversation_service.create_conversation(user_id)
    return [], new_conversation_id


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


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    agent: Agent = Depends(get_coach_agent),
    user_id: str | None = Depends(get_current_user_id),
) -> ChatResponse:
    if request.session_id:
        trace.get_current_span().set_attribute(SpanAttributes.SESSION_ID, request.session_id)

    message_history, conversation_id = await _resolve_context(request, user_id)

    try:
        result = await agent.run(request.message, message_history=message_history)
    except Exception as exc:
        logger.exception("Coach agent failed to produce a reply")
        raise HTTPException(status_code=502, detail=_AGENT_UNAVAILABLE_DETAIL) from exc

    await _persist_turn(user_id, conversation_id, request.message, result.output)
    return ChatResponse(reply=result.output, conversation_id=conversation_id)


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    agent: Agent = Depends(get_coach_agent),
    user_id: str | None = Depends(get_current_user_id),
) -> StreamingResponse:
    if request.session_id:
        trace.get_current_span().set_attribute(SpanAttributes.SESSION_ID, request.session_id)

    message_history, conversation_id = await _resolve_context(request, user_id)

    async def event_generator():
        chunks: list[str] = []
        try:
            if conversation_id:
                yield f"event: conversation\ndata: {json.dumps({'conversation_id': conversation_id})}\n\n"
            async with agent.run_stream(request.message, message_history=message_history) as result:
                async for delta in result.stream_text(delta=True):
                    chunks.append(delta)
                    yield f"data: {json.dumps({'delta': delta})}\n\n"
            await _persist_turn(user_id, conversation_id, request.message, "".join(chunks))
            yield "event: done\ndata: {}\n\n"
        except Exception:
            logger.exception("Coach agent failed to stream a reply")
            yield f"event: error\ndata: {json.dumps({'detail': _AGENT_UNAVAILABLE_DETAIL})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
