from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CostRecord:
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    task_type: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class CostTracker:
    """Tracks real-time cost, token usage, and performance across providers."""

    def __init__(self):
        self._records: list[CostRecord] = []
        self._by_provider: dict[str, list[CostRecord]] = defaultdict(list)
        self._by_model: dict[str, list[CostRecord]] = defaultdict(list)

    def record(self, rec: CostRecord) -> None:
        self._records.append(rec)
        self._by_provider[rec.provider].append(rec)
        self._by_model[rec.model].append(rec)

        if len(self._records) > 10_000:
            self._records = self._records[-5_000:]
            self._rebuild_indexes()

    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self._records)

    def total_tokens(self) -> dict[str, int]:
        prompt = sum(r.prompt_tokens for r in self._records)
        completion = sum(r.completion_tokens for r in self._records)
        return {"prompt": prompt, "completion": completion, "total": prompt + completion}

    def cost_by_provider(self) -> dict[str, float]:
        return {
            provider: sum(r.cost_usd for r in records)
            for provider, records in self._by_provider.items()
        }

    def cost_by_model(self) -> dict[str, float]:
        return {
            model: sum(r.cost_usd for r in records)
            for model, records in self._by_model.items()
        }

    def avg_latency_by_provider(self) -> dict[str, float]:
        result = {}
        for provider, records in self._by_provider.items():
            if records:
                result[provider] = round(
                    sum(r.latency_ms for r in records) / len(records), 2
                )
        return result

    def requests_per_provider(self) -> dict[str, int]:
        return {p: len(recs) for p, recs in self._by_provider.items()}

    def summary(self) -> dict[str, Any]:
        return {
            "total_requests": len(self._records),
            "total_cost_usd": round(self.total_cost(), 6),
            "total_tokens": self.total_tokens(),
            "cost_by_provider": {
                k: round(v, 6) for k, v in self.cost_by_provider().items()
            },
            "cost_by_model": {
                k: round(v, 6) for k, v in self.cost_by_model().items()
            },
            "avg_latency_by_provider": self.avg_latency_by_provider(),
            "requests_per_provider": self.requests_per_provider(),
        }

    def recent(self, n: int = 20) -> list[dict[str, Any]]:
        return [
            {
                "provider": r.provider,
                "model": r.model,
                "cost_usd": r.cost_usd,
                "latency_ms": r.latency_ms,
                "tokens": r.prompt_tokens + r.completion_tokens,
                "task_type": r.task_type,
                "timestamp": r.timestamp,
            }
            for r in self._records[-n:]
        ]

    def _rebuild_indexes(self) -> None:
        self._by_provider = defaultdict(list)
        self._by_model = defaultdict(list)
        for rec in self._records:
            self._by_provider[rec.provider].append(rec)
            self._by_model[rec.model].append(rec)
