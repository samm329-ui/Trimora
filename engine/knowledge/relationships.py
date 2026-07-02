import re
from typing import Optional

from ..data.models import Segment
from .local_graph import LocalKnowledgeGraph


_CONTRAST_WORDS = re.compile(r"(?i)\b(but|however|yet|actually|surprisingly|unexpectedly)\b")
_EXPLANATION_WORDS = re.compile(r"(?i)\b(because|meaning|which means|in other words|that is|specifically)\b")
_CONCLUSION_WORDS = re.compile(r"(?i)\b(so|therefore|thus|hence|in conclusion|ultimately|the point is)\b")
_EVIDENCE_WORDS = re.compile(r"(?i)\b(for example|for instance|such as|like|according to|studies show)\b")


def _determine_confidence(segment: Segment, pattern: Optional[re.Pattern] = None) -> float:
    base = 0.5
    if segment.sentiment != 0:
        base += abs(segment.sentiment) * 0.2
    if segment.patterns:
        base += min(len(segment.patterns) * 0.05, 0.2)
    if pattern and pattern.search(segment.text):
        base += 0.15
    return min(base, 0.95)


def _compute_edge_weight(source: Segment, target: Segment,
                          edge_type: str,
                          target_pos: float = 0.0) -> float:
    weight = 0.5
    if edge_type == "follows":
        gap = target.start - source.end
        if gap < 2:
            weight = 0.8
        elif gap < 10:
            weight = 0.6
        else:
            weight = 0.4
    elif edge_type == "contrasts":
        weight = 0.6 + abs(source.sentiment - target.sentiment) * 0.3
    elif edge_type == "explains":
        shared = len(set(source.patterns) & set(target.patterns))
        weight = 0.5 + shared * 0.1
    elif edge_type == "concludes":
        weight = 0.5 + target_pos * 0.3
    elif edge_type == "supports":
        shared = len(set(source.patterns) & set(target.patterns))
        weight = 0.4 + shared * 0.1 + abs(target.sentiment) * 0.1
    return min(weight, 1.0)


def _get_segment_position(segment: Segment, segments: list[Segment]) -> float:
    total_dur = max(s.end for s in segments) if segments else 1.0
    return segment.start / total_dur if total_dur > 0 else 0.0


def detect_relationships(
    segments: list[Segment],
    graph: LocalKnowledgeGraph,
    window_seconds: float = 120.0,
) -> None:
    for seg in segments:
        graph.add_segment(seg)

    for i, source in enumerate(segments):
        for j, target in enumerate(segments):
            if i >= j:
                continue
            gap = target.start - source.end
            if gap < 0 or gap > window_seconds:
                continue

            graph.add_edge(
                source.id, target.id,
                edge_type="follows",
                weight=_compute_edge_weight(source, target, "follows"),
                confidence=_determine_confidence(source),
            )

            if _CONTRAST_WORDS.search(target.text) and abs(source.sentiment - target.sentiment) > 0.2:
                graph.add_edge(
                    source.id, target.id,
                    edge_type="contrasts",
                    weight=_compute_edge_weight(source, target, "contrasts"),
                    confidence=_determine_confidence(target, _CONTRAST_WORDS),
                )

            shared_keywords = set(source.patterns) & set(target.patterns)
            if _EXPLANATION_WORDS.search(target.text) and shared_keywords:
                graph.add_edge(
                    source.id, target.id,
                    edge_type="explains",
                    weight=_compute_edge_weight(source, target, "explains"),
                    confidence=_determine_confidence(target, _EXPLANATION_WORDS),
                )

            target_pos = _get_segment_position(target, segments)
            if _CONCLUSION_WORDS.search(target.text) and target_pos > 0.6:
                graph.add_edge(
                    source.id, target.id,
                    edge_type="concludes",
                    weight=_compute_edge_weight(source, target, "concludes", target_pos),
                    confidence=_determine_confidence(target, _CONCLUSION_WORDS),
                )

            if _EVIDENCE_WORDS.search(target.text) or shared_keywords:
                graph.add_edge(
                    source.id, target.id,
                    edge_type="supports",
                    weight=_compute_edge_weight(source, target, "supports"),
                    confidence=_determine_confidence(target, _EVIDENCE_WORDS),
                )
