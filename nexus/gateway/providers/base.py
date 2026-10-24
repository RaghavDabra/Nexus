from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelInfo:
    provider: str
    model_id: str
    display_name: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    max_context: int
    strengths: list[str]
    latency_class: str  # "fast", "medium", "slow"


@dataclass
class CompletionRequest:
    messages: list[dict[str, str]]
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False
    metadata: dict[str, Any] | None = None


@dataclass
class CompletionResponse:
    content: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float
    metadata: dict[str, Any] | None = None


class BaseProvider(ABC):
    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...

    @abstractmethod
    def available_models(self) -> list[ModelInfo]:
        ...

    @abstractmethod
    def provider_name(self) -> str:
        ...
