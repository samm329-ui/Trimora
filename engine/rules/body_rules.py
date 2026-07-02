from ..data.models import BodyCandidate
from ..graph.knowledge_graph import KnowledgeGraph
from ..config import get_config


def find_body_candidates(hook_id: str, graph: KnowledgeGraph) -> list[BodyCandidate]:
    cfg = get_config().scoring
    rcfg = get_config().rule_scores
    hook_data = graph.graph.nodes.get(hook_id)
    if hook_data is None:
        return []

    hook_seg = hook_data.get("segment")
    if hook_seg is None:
        return []

    candidates = []

    for neighbor_id in graph.graph.successors(hook_id):
        edge_data = graph.graph.edges.get((hook_id, neighbor_id))
        if edge_data is None:
            continue
        neighbor_seg = graph.graph.nodes[neighbor_id].get("segment")
        if neighbor_seg is None:
            continue

        if edge_data["type"] in ("explains", "supports"):
            score = edge_data["weight"] * 100
            time_gap = neighbor_seg.start - hook_seg.end
            if 0 <= time_gap < cfg.TEMPORAL_WINDOW_SECONDS:
                score += rcfg.BODY_TEMPORAL_BONUS
            candidates.append(BodyCandidate(neighbor_id, score))

    for neighbor_id in graph.get_temporal_neighbors(hook_id, window=cfg.MATCH_WINDOW):
        if neighbor_id in [c.id for c in candidates]:
            continue
        neighbor_seg = graph.graph.nodes[neighbor_id].get("segment")
        if neighbor_seg is None:
            continue
        if neighbor_seg.start <= hook_seg.end:
            continue

        score = rcfg.BODY_BASELINE
        if "personal" in neighbor_seg.patterns:
            score += 15
        if abs(neighbor_seg.sentiment) > 0.2:
            score += 10
        candidates.append(BodyCandidate(neighbor_id, score))

    return sorted(candidates, key=lambda c: c.score, reverse=True)
