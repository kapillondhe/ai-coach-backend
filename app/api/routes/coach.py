from fastapi import APIRouter, Depends
from pydantic import BaseModel
from pydantic_ai import Agent

from app.agents.coach_agent import get_coach_agent

router = APIRouter(prefix="/coach", tags=["coach"])


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, agent: Agent = Depends(get_coach_agent)) -> ChatResponse:
    result = await agent.run(request.message)
    return ChatResponse(reply=result.output)
