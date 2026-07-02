import re
from ..data.models import Segment
from ..graph.knowledge_graph import KnowledgeGraph
from ..config import get_config

_EXPLAINS_PATTERNS = [
    r"(?i)which means",
    r"(?i)in other words",
    r"(?i)that is",
    r"(?i)i mean",
    r"(?i)what I (mean|m saying)",
]

_CONCLUDES_PATTERNS = [
    r"(?i)\bso\b",
    r"(?i)\btherefore\b",
    r"(?i)\bthat's why\b",
    r"(?i)\bin conclusion\b",
    r"(?i)\bto summarize?\b",
    r"(?i)\boverall\b",
]

_SUPPORTS_PATTERNS = [
    r"(?i)for example",
    r"(?i)like when",
    r"(?i)for instance",
    r"(?i)such as",
    r"(?i)to illustrate",
]


def detect_relationships(segments: list[Segment], graph: KnowledgeGraph):
    cfg = get_config().graph
    for i, seg in enumerate(segments):
        graph.add_segment(seg)

        if i > 0:
            graph.add_relationship(
                segments[i - 1].id, seg.id,
                "follows", cfg.FOLLOWS_WEIGHT,
                "temporal adjacency"
            )

        for j in range(i + 1, min(i + cfg.TEMPORAL_WINDOW_EDGES, len(segments))):
            other = segments[j]
            if not other.patterns:
                continue

            if "but" in seg.patterns and abs(seg.sentiment - other.sentiment) > get_config().scoring.SENTIMENT_CONTRAST_THRESHOLD:
                graph.add_relationship(
                    seg.id, other.id,
                    "contrasts", cfg.CONTRASTS_WEIGHT_MIN,
                    "opposing sentiment with contrast marker"
                )

            if _matches_explains(seg.text, other.text):
                graph.add_relationship(
                    seg.id, other.id,
                    "explains",
                    _compute_keyword_overlap_weight(seg.text, other.text,
                                                     cfg.EXPLAINS_WEIGHT_MIN, cfg.EXPLAINS_WEIGHT_MAX),
                    "shared keywords with explanation pattern"
                )

            if _matches_concludes(other.text, other.start / max(segments[-1].end, 1)):
                graph.add_relationship(
                    seg.id, other.id,
                    "concludes",
                    _compute_keyword_overlap_weight(seg.text, other.text,
                                                     cfg.CONCLUDES_WEIGHT_MIN, cfg.CONCLUDES_WEIGHT_MAX),
                    "conclusion pattern in later segment"
                )

            if _matches_supports(seg.text, other.text):
                graph.add_relationship(
                    seg.id, other.id,
                    "supports",
                    _compute_keyword_overlap_weight(seg.text, other.text,
                                                     cfg.SUPPORTS_WEIGHT_MIN, cfg.SUPPORTS_WEIGHT_MAX),
                    "support pattern with shared keywords"
                )


def _matches_explains(source_text: str, target_text: str) -> bool:
    shared = _shared_keywords(source_text, target_text)
    if shared < 2:
        return False
    for p in _EXPLAINS_PATTERNS:
        if re.search(p, target_text):
            return True
    return False


def _matches_concludes(text: str, position: float) -> bool:
    if position < 0.5:
        return False
    for p in _CONCLUDES_PATTERNS:
        if re.search(p, text):
            return True
    return False


def _matches_supports(source_text: str, target_text: str) -> bool:
    shared = _shared_keywords(source_text, target_text)
    if shared < 1:
        return False
    for p in _SUPPORTS_PATTERNS:
        if re.search(p, target_text):
            return True
    return False


def _shared_keywords(text1: str, text2: str) -> int:
    words1 = set(re.findall(r'\b[a-zA-Z]{4,}\b', text1.lower()))
    words2 = set(re.findall(r'\b[a-zA-Z]{4,}\b', text2.lower()))
    return len(words1 & words2)


def _compute_keyword_overlap_weight(text1: str, text2: str, min_w: float, max_w: float) -> float:
    shared = _shared_keywords(text1, text2)
    total = len(set(re.findall(r'\b[a-zA-Z]{4,}\b', text1.lower())) | set(re.findall(r'\b[a-zA-Z]{4,}\b', text2.lower())))
    if total == 0:
        return min_w
    ratio = shared / total
    return min_w + (max_w - min_w) * min(ratio * 3, 1.0)
