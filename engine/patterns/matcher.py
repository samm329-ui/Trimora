from typing import Optional

from .detector import DiscoveredPattern, MIN_PATTERN_LEN
from .meta import MetaPatternGraph


def compute_sequence_similarity(seq1: list[str], seq2: list[str]) -> float:
    shorter = min(len(seq1), len(seq2))
    if shorter == 0:
        return 0.0
    matches = sum(1 for a, b in zip(seq1[:shorter], seq2[:shorter]) if a == b)
    return matches / shorter


class PatternMatcher:
    def __init__(self):
        self._patterns: dict[str, DiscoveredPattern] = {}

    def register_pattern(self, pattern: DiscoveredPattern) -> None:
        key = "->".join(pattern.node_types)
        self._patterns[key] = pattern

    def match_against_global(
        self,
        local_sequence: list[str],
        category: str = "",
        meta_graph: Optional[MetaPatternGraph] = None,
    ) -> list[dict]:
        matches = []

        for i in range(len(local_sequence) - MIN_PATTERN_LEN + 1):
            for length in range(MIN_PATTERN_LEN, min(5, len(local_sequence) - i + 1)):
                subseq = local_sequence[i:i + length]
                key = "->".join(subseq)
                pattern = self._patterns.get(key)
                if pattern and pattern.confidence > 0.5:
                    matches.append({
                        "pattern_id": key,
                        "match_position": i,
                        "match_length": length,
                        "expected_saves": 0.0,
                        "expected_shares": 0.0,
                        "confidence": pattern.confidence,
                    })

        if category and meta_graph:
            preferred = meta_graph.get_preferred_structure(category)
            if preferred:
                similarity = compute_sequence_similarity(
                    local_sequence, preferred.node_types
                )
                if similarity > 0.6:
                    matches.append({
                        "pattern_id": "->".join(preferred.node_types),
                        "match_type": "meta",
                        "confidence": similarity * preferred.confidence,
                        "domain": category,
                    })

        matches.sort(key=lambda m: m.get("confidence", 0), reverse=True)
        return matches

    def find_similar(self, node_types: list[str], threshold: float = 0.5) -> list[dict]:
        results = []
        for key, pattern in self._patterns.items():
            similarity = compute_sequence_similarity(node_types, pattern.node_types)
            if similarity >= threshold:
                results.append({
                    "pattern": pattern,
                    "similarity": similarity,
                })
        results.sort(key=lambda r: r["similarity"], reverse=True)
        return results
