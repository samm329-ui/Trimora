from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone


@dataclass
class DiscoveredPattern:
    node_types: list[str]
    occurrences: int = 1
    confidence: float = 0.5
    avg_saves: float = 0.0
    avg_shares: float = 0.0
    first_seen: str = ""
    last_seen: str = ""

    def __post_init__(self):
        if not self.first_seen:
            now = datetime.now(timezone.utc).isoformat()
            self.first_seen = now
            self.last_seen = now

    def to_dict(self) -> dict:
        return {
            "node_types": self.node_types,
            "occurrences": self.occurrences,
            "confidence": self.confidence,
        }


MIN_PATTERN_LEN = 2
MAX_PATTERN_LEN = 5


class PatternDetector:
    def __init__(self):
        self._known_patterns: dict[str, DiscoveredPattern] = {}
        self._min_occurrences = 1

    def discover_patterns(self, sequence: list[str]) -> list[DiscoveredPattern]:
        discovered = []

        for length in range(MIN_PATTERN_LEN, min(MAX_PATTERN_LEN, len(sequence)) + 1):
            for i in range(len(sequence) - length + 1):
                subseq = sequence[i:i + length]
                key = "->".join(subseq)
                if key in self._known_patterns:
                    self._known_patterns[key].occurrences += 1
                else:
                    pattern = DiscoveredPattern(node_types=subseq)
                    self._known_patterns[key] = pattern

                discovered.append(self._known_patterns[key])

        for pattern in self._known_patterns.values():
            n = pattern.occurrences
            pattern.confidence = min(1.0 - (1.0 / (1.0 + n * 0.01)), 0.99)

        return discovered

    def find_pattern(self, node_types: list[str]) -> Optional[DiscoveredPattern]:
        key = "->".join(node_types)
        return self._known_patterns.get(key)

    def all_patterns(self) -> list[DiscoveredPattern]:
        return sorted(
            self._known_patterns.values(),
            key=lambda p: p.occurrences,
            reverse=True,
        )

    def pattern_count(self) -> int:
        return len(self._known_patterns)
