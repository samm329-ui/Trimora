from datetime import datetime, timezone
from typing import Optional

from .graph import PatternGraph
from .detector import DiscoveredPattern

BASE_DECAY_RATE = 0.0003
FRESHNESS_HALF_LIFE = 365


class PatternConfidence:
    def __init__(self):
        self._confidence_history: dict[str, list[dict]] = {}

    def compute_confidence(self, pattern: DiscoveredPattern) -> float:
        n = pattern.occurrences
        base_confidence = 1.0 - (1.0 / (1.0 + n * 0.01))
        return min(max(base_confidence, 0.01), 0.99)

    def compute_freshness(self, pattern: DiscoveredPattern) -> float:
        try:
            last_seen = datetime.fromisoformat(pattern.last_seen)
        except (ValueError, TypeError):
            return 0.5
        days_since = (datetime.now(timezone.utc) - last_seen).days
        return 2.0 ** (-days_since / FRESHNESS_HALF_LIFE)

    def apply_decay(self, pattern: DiscoveredPattern) -> float:
        try:
            last_seen = datetime.fromisoformat(pattern.last_seen)
        except (ValueError, TypeError):
            return pattern.confidence
        days_since = (datetime.now(timezone.utc) - last_seen).days
        decay = BASE_DECAY_RATE * days_since
        return max(pattern.confidence - decay, 0.01)

    def boost_on_success(self, pattern: DiscoveredPattern,
                          actual_saves: float, actual_shares: float) -> None:
        expected = 0.5
        actual = (actual_saves + actual_shares) / 2
        if actual > expected * 1.1:
            pattern.confidence = min(pattern.confidence + 0.02, 0.99)
            now = datetime.now(timezone.utc).isoformat()
            pattern.last_seen = now
            self._log_confidence(pattern, "performance_boost")

    def refresh_all(self, graph: PatternGraph) -> None:
        for pattern in graph.top_patterns(100):
            pattern.confidence = self.apply_decay(pattern)

    def _log_confidence(self, pattern: DiscoveredPattern, trigger: str) -> None:
        key = "->".join(pattern.node_types)
        if key not in self._confidence_history:
            self._confidence_history[key] = []
        self._confidence_history[key].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence": pattern.confidence,
            "trigger": trigger,
        })

    def get_confidence_history(self, pattern: DiscoveredPattern) -> list[dict]:
        key = "->".join(pattern.node_types)
        return self._confidence_history.get(key, [])
