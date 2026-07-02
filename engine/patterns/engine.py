from typing import Optional

from .detector import PatternDetector
from .matcher import PatternMatcher
from .graph import PatternGraph
from .context import PatternContext
from .confidence import PatternConfidence
from .meta import MetaPatternGraph
from ..data.models import Segment
from ..knowledge.global_graph import GlobalKnowledgeGraph
from ..knowledge.context_db import ContextDatabase


class PatternIntelligenceEngine:
    def __init__(self, global_graph: Optional[GlobalKnowledgeGraph] = None,
                 context_db: Optional[ContextDatabase] = None):
        self.detector = PatternDetector()
        self.matcher = PatternMatcher()
        self.graph = PatternGraph()
        self.context = PatternContext(context_db or ContextDatabase())
        self.confidence = PatternConfidence()
        self.meta = MetaPatternGraph(global_graph or GlobalKnowledgeGraph())

        self.patterns_discovered: list[dict] = []
        self.current_video_patterns: list[dict] = []

    def process_video(self, segments: list[Segment],
                       labels: Optional[list[dict]] = None,
                       category: str = "") -> dict:
        self.current_video_patterns = []
        label_data = labels or []

        local_sequence = self._extract_label_sequence(segments, label_data)

        discovered = self.detector.discover_patterns(local_sequence)
        for pattern in discovered:
            self._register_pattern(pattern, segments, label_data)
            self.current_video_patterns.append(pattern)

        matches = self.matcher.match_against_global(local_sequence, category, self.meta)
        self._update_from_matches(matches)

        self.confidence.refresh_all(self.graph)

        return {
            "video_patterns": len(discovered),
            "global_matches": len(matches),
            "active_patterns": self.graph.pattern_count(),
            "patterns": [p.to_dict() for p in discovered] if discovered else [],
        }

    def _extract_label_sequence(self, segments: list[Segment],
                                 labels: list[dict]) -> list[str]:
        if labels:
            return [
                lb.get("emotion", "neutral") if isinstance(lb, dict) else "neutral"
                for lb in labels
            ]
        return [seg.emotion if hasattr(seg, 'emotion') and seg.emotion else "neutral"
                for seg in segments]

    def _register_pattern(self, pattern, segments: list[Segment],
                           labels: list[dict]) -> None:
        self.graph.add_pattern(pattern)
        for node_type in pattern.node_types:
            matched_segs = [s for s in segments if any(p == node_type for p in s.patterns)]
            for seg in matched_segs:
                self.context.register_segment_context(node_type, seg.text)
                self.context.record_pattern_context(node_type, seg.text)

    def _update_from_matches(self, matches: list[dict]) -> None:
        self.patterns_discovered.extend(matches)

    def get_statistics(self) -> dict:
        return {
            "patterns_discovered": len(self.patterns_discovered),
            "graph_pattern_count": self.graph.pattern_count(),
            "context_types_count": self.context.type_count(),
        }
