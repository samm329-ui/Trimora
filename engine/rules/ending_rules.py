from ..data.models import EndingCandidate
from ..graph.knowledge_graph import KnowledgeGraph
from ..config import get_config


def find_ending_candidates(hook_id: str, body_ids: list[str],
                           graph: KnowledgeGraph) -> list[EndingCandidate]:
    cfg = get_config().scoring
    rcfg = get_config().rule_scores

    hook_data = graph.graph.nodes.get(hook_id)
    if hook_data is None:
        return []
    hook_seg = hook_data.get("segment")
    if hook_seg is None:
        return []

    last_body = None
    for bid in reversed(body_ids):
        bd = graph.graph.nodes.get(bid)
        if bd and bd.get("segment"):
            last_body = bd["segment"]
            break

    if last_body is None:
        last_body = hook_seg

    candidates = []

    for node_id, data in graph.graph.nodes(data=True):
        seg = data.get("segment")
        if seg is None:
            continue

        if seg.start <= last_body.start:
            continue

        score = 0.0

        if seg.sentiment > 0.2:
            score += rcfg.ENDING_POSITIVE

        takeaway_patterns = {"key_lesson", "action", "remember", "point", "call_to_action", "here_is"}
        if any(p in seg.patterns for p in takeaway_patterns):
            score += rcfg.ENDING_TAKEAWAY

        summary_patterns = {"finally", "so", "therefore", "conclusion"}
        if any(p in seg.patterns for p in summary_patterns):
            score += rcfg.ENDING_SUMMARY

        if cfg.ENDING_MIN_DURATION <= seg.duration <= cfg.ENDING_MAX_DURATION:
            score += rcfg.ENDING_DURATION_FIT
        elif cfg.ENDING_LOOSE_MIN <= seg.duration <= cfg.ENDING_LOOSE_MAX:
            score += rcfg.ENDING_DURATION_LOOSE

        if seg.start / max(graph.graph.nodes[node_id].get("position", seg.start), 1) > cfg.RECENCY_THRESHOLD \
           or (hasattr(seg, 'start') and seg.start / max(hook_seg.start + 1, 1) > 0.7):
            score += rcfg.ENDING_RECENCY

        practicality_patterns = {"steps", "lesson", "framework", "tip", "rule", "checklist", "method"}
        if any(p in seg.patterns for p in practicality_patterns):
            score += rcfg.ENDING_PRACTICALITY

        if "personal" in seg.patterns:
            score += rcfg.ENDING_RELATABLE

        if seg.sentiment > 0.3 and \
           any(p in seg.patterns for p in {"key_lesson", "action", "point", "call_to_action", "here_is"}) and \
           "personal" in seg.patterns:
            score += rcfg.ENDING_RESOLUTION_BONUS

        if score > cfg.ENDING_MIN_SCORE:
            candidates.append(EndingCandidate(node_id, score))

    return sorted(candidates, key=lambda c: c.score, reverse=True)
