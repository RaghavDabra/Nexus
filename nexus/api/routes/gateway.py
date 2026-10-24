from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/gateway", tags=["gateway"])

_orchestrator = None
_router_instance = None


def init_gateway_routes(orchestrator, router_instance):
    global _orchestrator, _router_instance
    _orchestrator = orchestrator
    _router_instance = router_instance


class CompletionRequestBody(BaseModel):
    messages: list[dict[str, str]]
    strategy: str = "balanced"
    temperature: float = 0.7
    max_tokens: int = 4096
    provider: str | None = None
    model: str | None = None


class CompletionResponseBody(BaseModel):
    request_id: str
    status: str
    result: dict[str, Any] | None = None
    decisions: list[dict[str, Any]] = []
    orchestration_ms: float = 0


@router.post("/completions", response_model=CompletionResponseBody)
async def create_completion(body: CompletionRequestBody):
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    payload: dict[str, Any] = {
        "messages": body.messages,
        "strategy": body.strategy,
        "temperature": body.temperature,
        "max_tokens": body.max_tokens,
    }

    if body.provider:
        payload["force_provider"] = body.provider
    if body.model:
        payload["force_model"] = body.model

    result = await _orchestrator.process_request("gateway.request", payload)

    return CompletionResponseBody(
        request_id=result.get("request_id", ""),
        status=result.get("status", "unknown"),
        result=result.get("result"),
        decisions=result.get("decisions", []),
        orchestration_ms=result.get("orchestration_ms", 0),
    )


@router.get("/models")
async def list_models():
    if not _router_instance:
        raise HTTPException(status_code=503, detail="Router not initialized")
    models = _router_instance.all_models()
    return {
        "models": [
            {
                "provider": m.provider,
                "model_id": m.model_id,
                "display_name": m.display_name,
                "input_cost_per_1k": m.input_cost_per_1k,
                "output_cost_per_1k": m.output_cost_per_1k,
                "max_context": m.max_context,
                "strengths": m.strengths,
                "latency_class": m.latency_class,
            }
            for m in models
        ]
    }
