from __future__ import annotations

import asyncio
import random
import time

from nexus.gateway.providers.base import (
    BaseProvider,
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
)

MOCK_MODELS = [
    ModelInfo(
        provider="mock",
        model_id="mock-fast",
        display_name="Mock Fast",
        input_cost_per_1k=0.0001,
        output_cost_per_1k=0.0002,
        max_context=32_000,
        strengths=["fast", "cheap", "testing"],
        latency_class="fast",
    ),
    ModelInfo(
        provider="mock",
        model_id="mock-quality",
        display_name="Mock Quality",
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.002,
        max_context=128_000,
        strengths=["reasoning", "code", "testing"],
        latency_class="medium",
    ),
]


class MockProvider(BaseProvider):
    """Mock provider for testing and demos without API keys."""

    def __init__(self, failure_rate: float = 0.0):
        self._failure_rate = failure_rate

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        model = request.model or "mock-fast"
        start = time.monotonic()

        latency = random.uniform(50, 200) if model == "mock-fast" else random.uniform(200, 800)
        await asyncio.sleep(latency / 1000)

        if random.random() < self._failure_rate:
            raise RuntimeError("Mock provider simulated failure")

        prompt_text = " ".join(m["content"] for m in request.messages)
        prompt_tokens = len(prompt_text.split()) * 2
        completion_tokens = random.randint(50, 300)

        model_info = self._find_model(model)
        cost = (
            (prompt_tokens / 1000) * model_info.input_cost_per_1k
            + (completion_tokens / 1000) * model_info.output_cost_per_1k
        )

        elapsed_ms = (time.monotonic() - start) * 1000

        return CompletionResponse(
            content=f"[Mock response from {model}] Processed {len(request.messages)} messages. "
            f"This is a simulated response for development and testing.",
            model=model,
            provider="mock",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(elapsed_ms, 2),
            cost_usd=round(cost, 6),
        )

    async def health_check(self) -> bool:
        return random.random() > self._failure_rate

    def available_models(self) -> list[ModelInfo]:
        return MOCK_MODELS

    def provider_name(self) -> str:
        return "mock"

    def _find_model(self, model_id: str) -> ModelInfo:
        for m in MOCK_MODELS:
            if m.model_id == model_id:
                return m
        return MOCK_MODELS[0]
