from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class PredictionResult:
    device_id: int
    predicted_utilization: float
    confidence: float
    trend: str  # "rising", "falling", "stable", "volatile"
    time_horizon_seconds: int
    will_overload: bool
    will_idle: bool
    recommended_action: str | None


class WorkloadPredictor:
    """Predicts GPU workload patterns using exponential smoothing and trend analysis.

    Maintains a sliding window of historical metrics per device and forecasts
    future utilization to enable proactive scheduling decisions.
    """

    def __init__(self, window_size: int = 120, alpha: float = 0.3):
        self._window_size = window_size
        self._alpha = alpha
        self._history: dict[int, deque[float]] = {}
        self._smoothed: dict[int, float] = {}
        self._trend: dict[int, float] = {}

    def record(self, device_id: int, utilization: float) -> None:
        if device_id not in self._history:
            self._history[device_id] = deque(maxlen=self._window_size)
            self._smoothed[device_id] = utilization
            self._trend[device_id] = 0

        self._history[device_id].append(utilization)
        prev_smoothed = self._smoothed[device_id]
        self._smoothed[device_id] = self._alpha * utilization + (1 - self._alpha) * prev_smoothed
        self._trend[device_id] = (
            self._alpha * (self._smoothed[device_id] - prev_smoothed)
            + (1 - self._alpha) * self._trend[device_id]
        )

    def predict(self, device_id: int, steps_ahead: int = 10) -> PredictionResult:
        if device_id not in self._history or len(self._history[device_id]) < 3:
            return PredictionResult(
                device_id=device_id,
                predicted_utilization=50.0,
                confidence=0.0,
                trend="unknown",
                time_horizon_seconds=steps_ahead * 10,
                will_overload=False,
                will_idle=False,
                recommended_action=None,
            )

        forecast = self._smoothed[device_id] + steps_ahead * self._trend[device_id]
        forecast = max(0, min(100, forecast))

        history = list(self._history[device_id])
        volatility = self._compute_volatility(history)
        confidence = max(0, min(1, 1 - volatility / 50))
        trend = self._classify_trend(history, self._trend[device_id], volatility)

        will_overload = forecast > 85
        will_idle = forecast < 5

        action = None
        if will_overload:
            action = "preemptive_migration"
        elif will_idle:
            action = "consolidate_or_scale_down"
        elif trend == "rising" and forecast > 70:
            action = "prepare_overflow_capacity"
        elif trend == "volatile":
            action = "increase_monitoring_frequency"

        return PredictionResult(
            device_id=device_id,
            predicted_utilization=round(forecast, 1),
            confidence=round(confidence, 3),
            trend=trend,
            time_horizon_seconds=steps_ahead * 10,
            will_overload=will_overload,
            will_idle=will_idle,
            recommended_action=action,
        )

    def predict_all(self, steps_ahead: int = 10) -> list[PredictionResult]:
        return [self.predict(did, steps_ahead) for did in self._history]

    def anomaly_score(self, device_id: int) -> float:
        if device_id not in self._history or len(self._history[device_id]) < 10:
            return 0.0

        history = list(self._history[device_id])
        current = history[-1]
        mean = sum(history) / len(history)
        std = self._std_dev(history)

        if std == 0:
            return 0.0
        return min(1.0, abs(current - mean) / (3 * std))

    def summary(self) -> dict[str, Any]:
        results = {}
        for device_id in self._history:
            pred = self.predict(device_id)
            results[str(device_id)] = {
                "current_smoothed": round(self._smoothed.get(device_id, 0), 1),
                "trend_direction": pred.trend,
                "predicted_utilization": pred.predicted_utilization,
                "confidence": pred.confidence,
                "anomaly_score": round(self.anomaly_score(device_id), 3),
                "data_points": len(self._history[device_id]),
            }
        return results

    def _compute_volatility(self, history: list[float]) -> float:
        if len(history) < 2:
            return 0
        diffs = [abs(history[i] - history[i - 1]) for i in range(1, len(history))]
        return sum(diffs) / len(diffs)

    def _classify_trend(self, history: list[float], trend_value: float, volatility: float) -> str:
        if volatility > 20:
            return "volatile"
        if abs(trend_value) < 0.5:
            return "stable"
        return "rising" if trend_value > 0 else "falling"

    def _std_dev(self, values: list[float]) -> float:
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
