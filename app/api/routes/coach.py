import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from pydantic import BaseModel
from pydantic_ai import Agent

from app.agents.coach_agent import get_coach_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/coach", tags=["coach"])

_AGENT_UNAVAILABLE_DETAIL = "Coach agent is temporarily unavailable. Please try again."


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, agent: Agent = Depends(get_coach_agent)) -> ChatResponse:
    if request.session_id:
        trace.get_current_span().set_attribute(SpanAttributes.SESSION_ID, request.session_id)

    try:
        result = await agent.run(request.message)
    except Exception:
        logger.exception("Coach agent failed to produce a reply")
        raise HTTPException(status_code=502, detail=_AGENT_UNAVAILABLE_DETAIL)
    return ChatResponse(reply=result.output)


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, agent: Agent = Depends(get_coach_agent)) -> StreamingResponse:
    if request.session_id:
        trace.get_current_span().set_attribute(SpanAttributes.SESSION_ID, request.session_id)

    async def event_generator():
        try:
            async with agent.run_stream(request.message) as result:
                async for delta in result.stream_text(delta=True):
                    yield f"data: {json.dumps({'delta': delta})}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception:
            logger.exception("Coach agent failed to stream a reply")
            yield f"event: error\ndata: {json.dumps({'detail': _AGENT_UNAVAILABLE_DETAIL})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
