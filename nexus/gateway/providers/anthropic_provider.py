from __future__ import annotations

import time

import httpx

from nexus.gateway.providers.base import (
    BaseProvider,
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
)


ANTHROPIC_MODELS = [
    ModelInfo(
        provider="anthropic",
        model_id="claude-sonnet-4-20250514",
        display_name="Claude Sonnet 4",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        max_context=200_000,
        strengths=["reasoning", "code", "analysis", "long-context"],
        latency_class="medium",
    ),
    ModelInfo(
        provider="anthropic",
        model_id="claude-haiku-3-5-20241022",
        display_name="Claude Haiku 3.5",
        input_cost_per_1k=0.0008,
        output_cost_per_1k=0.004,
        max_context=200_000,
        strengths=["fast", "cheap", "general"],
        latency_class="fast",
    ),
]


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url="https://api.anthropic.com/v1",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=60.0,
        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        model = request.model or "claude-haiku-3-5-20241022"
        start = time.monotonic()

        messages = []
        system_msg = None
        for msg in request.messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                messages.append(msg)

        body: dict = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if system_msg:
            body["system"] = system_msg

        resp = await self._client.post("/messages", json=body)
        resp.raise_for_status()
        data = resp.json()

        elapsed_ms = (time.monotonic() - start) * 1000
        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        model_info = self._find_model(model)
        cost = (
            (prompt_tokens / 1000) * model_info.input_cost_per_1k
            + (completion_tokens / 1000) * model_info.output_cost_per_1k
        )

        content_blocks = data.get("content", [])
        text = "".join(b["text"] for b in content_blocks if b["type"] == "text")

        return CompletionResponse(
            content=text,
            model=model,
            provider="anthropic",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(elapsed_ms, 2),
            cost_usd=round(cost, 6),
        )

    async def health_check(self) -> bool:
        try:
            resp = await self._client.post(
                "/messages",
                json={
                    "model": "claude-haiku-3-5-20241022",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                },
            )
            return resp.status_code == 200
        except Exception:
            return False

    def available_models(self) -> list[ModelInfo]:
        return ANTHROPIC_MODELS

    def provider_name(self) -> str:
        return "anthropic"

    def _find_model(self, model_id: str) -> ModelInfo:
        for m in ANTHROPIC_MODELS:
            if m.model_id == model_id:
                return m
        return ANTHROPIC_MODELS[0]
