import logging

from fastapi import APIRouter, Depends, HTTPException
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from pydantic import BaseModel
from pydantic_ai import Agent

from app.agents.coach_agent import get_coach_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/coach", tags=["coach"])


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
        raise HTTPException(status_code=502, detail="Coach agent is temporarily unavailable. Please try again.")
    return ChatResponse(reply=result.output)
