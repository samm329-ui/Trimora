from collections import defaultdict
from typing import Optional

from ..knowledge.global_graph import GlobalKnowledgeGraph
from .detector import DiscoveredPattern


class MetaPatternGraph:
    def __init__(self, global_graph: Optional[GlobalKnowledgeGraph] = None):
        self._gg = global_graph or GlobalKnowledgeGraph()
        self._category_patterns: dict[str, list[dict]] = defaultdict(list)

    def register_pattern_for_category(self, pattern: DiscoveredPattern,
                                       category: str, avg_saves: float = 0.0,
                                       avg_shares: float = 0.0) -> None:
        self._category_patterns[category].append({
            "node_types": pattern.node_types,
            "occurrences": pattern.occurrences,
            "confidence": pattern.confidence,
            "avg_saves": avg_saves,
            "avg_shares": avg_shares,
        })

    def get_preferred_structure(self, category: str) -> Optional[DiscoveredPattern]:
        candidates = self._category_patterns.get(category, [])
        if not candidates:
            return None
        best = max(candidates, key=lambda c: c.get("avg_saves", 0) * c.get("confidence", 0))
        return DiscoveredPattern(
            node_types=best["node_types"],
            occurrences=best["occurrences"],
            confidence=best["confidence"],
        )

    def get_edge_weight(self, source_type: str, target_type: str,
                         category: str = "") -> float:
        edge = self._gg.get_edge(source_type, target_type)
        if edge:
            base = edge.get("transition_probability" if category else "avg_confidence", 0.5)
            return base if isinstance(base, (int, float)) else 0.5
        return 0.5

    def get_preferred_hook(self, category: str) -> Optional[str]:
        candidates = self._category_patterns.get(category, [])
        if not candidates:
            return None
        first_nodes = [c["node_types"][0] for c in candidates if c["node_types"]]
        if not first_nodes:
            return None
        from collections import Counter
        return Counter(first_nodes).most_common(1)[0][0]

    def category_count(self) -> int:
        return len(self._category_patterns)

    def get_categories(self) -> list[str]:
        return list(self._category_patterns.keys())
