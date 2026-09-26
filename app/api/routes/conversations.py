import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.auth import require_user_id
from app.services import conversation as conversation_service
from app.services.conversation import ConversationNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])


class ConversationSummaryResponse(BaseModel):
    id: str
    title: str | None
    updated_at: str


class ConversationMessageResponse(BaseModel):
    role: str
    content: str


class ConversationDetailResponse(BaseModel):
    id: str
    title: str | None
    messages: list[ConversationMessageResponse]


@router.get("")
async def list_conversations(user_id: str = Depends(require_user_id)) -> list[ConversationSummaryResponse]:
    summaries = await conversation_service.list_conversations(user_id)
    return [
        ConversationSummaryResponse(id=s.id, title=s.title, updated_at=s.updated_at.isoformat())
        for s in summaries
    ]


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: str, user_id: str = Depends(require_user_id)
) -> ConversationDetailResponse:
    try:
        data = await conversation_service.get_conversation(conversation_id, user_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    return ConversationDetailResponse(
        id=data.id,
        title=data.title,
        messages=[ConversationMessageResponse(role=m.role, content=m.content) for m in data.messages],
    )


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str, user_id: str = Depends(require_user_id)
) -> dict[str, bool]:
    deleted = await conversation_service.delete_conversation(conversation_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True}
