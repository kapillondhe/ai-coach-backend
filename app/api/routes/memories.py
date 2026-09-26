import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.auth import require_user_id
from app.services import memory as memory_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memories", tags=["memories"])


class MemoryResponse(BaseModel):
    id: str
    content: str
    created_at: str


@router.get("")
async def list_memories(user_id: str = Depends(require_user_id)) -> list[MemoryResponse]:
    memories = await memory_service.list_memories(user_id)
    return [
        MemoryResponse(id=m.id, content=m.content, created_at=m.created_at.isoformat())
        for m in memories
    ]


@router.delete("/{memory_id}")
async def delete_memory(memory_id: str, user_id: str = Depends(require_user_id)) -> dict[str, bool]:
    deleted = await memory_service.forget(user_id, memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted": True}
