from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])

_cost_tracker = None
_health_agent = None


def init_analytics_routes(cost_tracker, health_agent):
    global _cost_tracker, _health_agent
    _cost_tracker = cost_tracker
    _health_agent = health_agent


@router.get("/costs")
async def cost_summary():
    if not _cost_tracker:
        raise HTTPException(status_code=503, detail="Cost tracker not initialized")
    return _cost_tracker.summary()


@router.get("/costs/recent")
async def recent_costs(limit: int = 20):
    if not _cost_tracker:
        raise HTTPException(status_code=503, detail="Cost tracker not initialized")
    return {"records": _cost_tracker.recent(limit)}


@router.get("/health")
async def health_summary():
    if not _health_agent:
        raise HTTPException(status_code=503, detail="Health agent not initialized")
    return {"providers": _health_agent.health_summary()}
