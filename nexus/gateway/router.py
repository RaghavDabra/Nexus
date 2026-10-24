from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from nexus.gateway.providers.base import BaseProvider, CompletionRequest, CompletionResponse, ModelInfo

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    provider: str
    model: str
    reasoning: str
    confidence: float
    cost_estimate: float
    latency_estimate: str


class TaskClassifier:
    """Classifies incoming requests by task type to inform routing."""

    TASK_KEYWORDS: dict[str, list[str]] = {
        "code": ["code", "function", "debug", "implement", "refactor", "python", "javascript", "api", "bug", "error"],
        "reasoning": ["analyze", "explain", "why", "compare", "evaluate", "think", "reason", "strategy"],
        "creative": ["write", "story", "poem", "creative", "generate", "imagine", "design"],
        "extraction": ["extract", "summarize", "list", "parse", "find", "search", "data"],
        "conversation": ["hello", "hi", "thanks", "help", "chat"],
    }

    def classify(self, messages: list[dict[str, str]]) -> str:
        text = " ".join(m["content"].lower() for m in messages)
        scores: dict[str, int] = {}
        for task_type, keywords in self.TASK_KEYWORDS.items():
            scores[task_type] = sum(1 for kw in keywords if kw in text)
        if not any(scores.values()):
            return "general"
        return max(scores, key=scores.get)  # type: ignore[arg-type]


TASK_MODEL_AFFINITY: dict[str, list[str]] = {
    "code": ["reasoning", "code"],
    "reasoning": ["reasoning", "analysis", "long-context"],
    "creative": ["general", "reasoning"],
    "extraction": ["fast", "cheap"],
    "conversation": ["fast", "cheap"],
    "general": ["general"],
}


class Router:
    """Intelligent model router.

    Routes requests to the optimal provider/model based on:
    - Task type classification
    - Model strengths and cost
    - Provider health status
    - Latency requirements
    """

    def __init__(self):
        self.providers: dict[str, BaseProvider] = {}
        self.classifier = TaskClassifier()
        self._health_cache: dict[str, bool] = {}

    def register_provider(self, provider: BaseProvider) -> None:
        self.providers[provider.provider_name()] = provider

    def all_models(self) -> list[ModelInfo]:
        models = []
        for provider in self.providers.values():
            models.extend(provider.available_models())
        return models

    def route(
        self,
        request: CompletionRequest,
        strategy: str = "balanced",
        health_status: dict[str, bool] | None = None,
    ) -> RoutingDecision:
        health = health_status or self._health_cache
        task_type = self.classifier.classify(request.messages)
        desired_strengths = TASK_MODEL_AFFINITY.get(task_type, ["general"])

        candidates = []
        for model in self.all_models():
            provider_key = f"{model.provider}:{model.model_id}"
            if health.get(provider_key) is False:
                continue
            score = self._score_model(model, desired_strengths, strategy)
            candidates.append((model, score))

        candidates.sort(key=lambda x: x[1], reverse=True)

        if not candidates:
            raise RuntimeError("No healthy models available for routing")

        best_model, score = candidates[0]
        prompt_tokens_est = sum(len(m["content"].split()) * 2 for m in request.messages)
        cost_est = (prompt_tokens_est / 1000) * best_model.input_cost_per_1k

        return RoutingDecision(
            provider=best_model.provider,
            model=best_model.model_id,
            reasoning=f"Task={task_type}, strategy={strategy}, strengths={desired_strengths}",
            confidence=min(score / 10, 1.0),
            cost_estimate=round(cost_est, 6),
            latency_estimate=best_model.latency_class,
        )

    async def execute(
        self,
        request: CompletionRequest,
        decision: RoutingDecision,
    ) -> CompletionResponse:
        provider = self.providers.get(decision.provider)
        if not provider:
            raise RuntimeError(f"Provider {decision.provider} not registered")

        request.model = decision.model
        return await provider.complete(request)

    def _score_model(
        self, model: ModelInfo, desired_strengths: list[str], strategy: str
    ) -> float:
        strength_overlap = len(set(model.strengths) & set(desired_strengths))
        score = strength_overlap * 3.0

        if strategy == "cost":
            score += (1.0 / (model.input_cost_per_1k + model.output_cost_per_1k + 0.001)) * 0.5
        elif strategy == "quality":
            score += strength_overlap * 2.0
        elif strategy == "speed":
            latency_bonus = {"fast": 5, "medium": 2, "slow": 0}
            score += latency_bonus.get(model.latency_class, 0)
        else:
            score += (1.0 / (model.input_cost_per_1k + model.output_cost_per_1k + 0.001)) * 0.2
            latency_bonus = {"fast": 2, "medium": 1, "slow": 0}
            score += latency_bonus.get(model.latency_class, 0)

        return score

    @staticmethod
    def hash_prompt(messages: list[dict[str, str]]) -> str:
        content = "|".join(f"{m['role']}:{m['content']}" for m in messages)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
