from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.auth import require_user_id
from app.services import dashboard as dashboard_service

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


class SparklinePointResponse(BaseModel):
    date: str
    value: float | None


class TrendCardResponse(BaseModel):
    id: str
    label: str
    sparkline: list[SparklinePointResponse]
    takeaway: str
    latest_value: float | None = None
    extra: dict = {}


class DashboardSummaryResponse(BaseModel):
    connected: bool
    syncing: bool = False
    trends: list[TrendCardResponse] = []


@router.get("/summary")
async def get_dashboard_summary(user_id: str = Depends(require_user_id)) -> DashboardSummaryResponse:
    summary = await dashboard_service.get_dashboard_summary(user_id)
    return DashboardSummaryResponse(
        connected=summary.connected,
        syncing=summary.syncing,
        trends=[
            TrendCardResponse(
                id=t.id,
                label=t.label,
                sparkline=[SparklinePointResponse(date=p.date, value=p.value) for p in t.sparkline],
                takeaway=t.takeaway,
                latest_value=t.latest_value,
                extra=t.extra,
            )
            for t in summary.trends
        ],
    )
