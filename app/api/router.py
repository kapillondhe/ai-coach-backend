from fastapi import APIRouter

from app.api.routes import coach, health, integrations, profile

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(coach.router)
api_router.include_router(integrations.router)
api_router.include_router(profile.router)
