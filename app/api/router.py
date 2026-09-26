from fastapi import APIRouter

from app.api.routes import coach, conversations, dashboard, health, integrations, memories, profile, sync

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(coach.router)
api_router.include_router(integrations.router)
api_router.include_router(profile.router)
api_router.include_router(conversations.router)
api_router.include_router(memories.router)
api_router.include_router(sync.router)
api_router.include_router(dashboard.router)
