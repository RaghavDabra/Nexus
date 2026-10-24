from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])

_orchestrator = None


def init_agent_routes(orchestrator):
    global _orchestrator
    _orchestrator = orchestrator


@router.get("/status")
async def agent_statuses():
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return {"agents": _orchestrator.agent_statuses()}


@router.get("/{agent_id}/status")
async def agent_status(agent_id: str):
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    agent = _orchestrator.agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    return agent.status()


@router.get("/{agent_id}/decisions")
async def agent_decisions(agent_id: str, limit: int = 20):
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    agent = _orchestrator.agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    return {"decisions": agent.metrics.decisions[-limit:]}
