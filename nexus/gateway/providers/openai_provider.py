from __future__ import annotations

import time
from typing import Any

import httpx

from nexus.gateway.providers.base import (
    BaseProvider,
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
)


OPENAI_MODELS = [
    ModelInfo(
        provider="openai",
        model_id="gpt-4o",
        display_name="GPT-4o",
        input_cost_per_1k=0.0025,
        output_cost_per_1k=0.01,
        max_context=128_000,
        strengths=["general", "reasoning", "code", "vision"],
        latency_class="medium",
    ),
    ModelInfo(
        provider="openai",
        model_id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
        max_context=128_000,
        strengths=["general", "fast", "cheap"],
        latency_class="fast",
    ),
]


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        model = request.model or "gpt-4o-mini"
        start = time.monotonic()

        resp = await self._client.post(
            "/chat/completions",
            json={
                "model": model,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        elapsed_ms = (time.monotonic() - start) * 1000
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        model_info = self._find_model(model)
        cost = (
            (prompt_tokens / 1000) * model_info.input_cost_per_1k
            + (completion_tokens / 1000) * model_info.output_cost_per_1k
        )

        return CompletionResponse(
            content=data["choices"][0]["message"]["content"],
            model=model,
            provider="openai",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(elapsed_ms, 2),
            cost_usd=round(cost, 6),
        )

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/models")
            return resp.status_code == 200
        except Exception:
            return False

    def available_models(self) -> list[ModelInfo]:
        return OPENAI_MODELS

    def provider_name(self) -> str:
        return "openai"

    def _find_model(self, model_id: str) -> ModelInfo:
        for m in OPENAI_MODELS:
            if m.model_id == model_id:
                return m
        return OPENAI_MODELS[0]
