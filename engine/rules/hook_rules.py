from ..data.models import HookCandidate
from ..graph.knowledge_graph import KnowledgeGraph
from ..config import get_config


def find_hook_candidates(graph: KnowledgeGraph) -> list[HookCandidate]:
    cfg = get_config().scoring
    rcfg = get_config().rule_scores
    candidates = []

    for node_id, data in graph.graph.nodes(data=True):
        seg = data.get("segment")
        if seg is None:
            continue

        if not (cfg.HOOK_MIN_DURATION <= seg.duration <= cfg.HOOK_MAX_DURATION):
            continue

        score = 0.0
        matched_rules = []

        if seg.speech_rate > cfg.HOOK_SPEECH_RATE_MIN:
            score += rcfg.HOOK_ENERGY_SPEECH
            matched_rules.append("high_speech_rate")

        if seg.volume_delta > cfg.HOOK_VOLUME_DELTA_MIN:
            score += rcfg.HOOK_ENERGY_VOLUME
            matched_rules.append("high_volume")

        curiosity_patterns = {"what_if", "unknown", "biggest", "question", "unexpected", "imagine"}
        if any(p in seg.patterns for p in curiosity_patterns):
            score += rcfg.HOOK_CURIOSITY
            matched_rules.append("curiosity")

        if seg.sentiment < -0.2 and "personal" in seg.patterns:
            score += rcfg.HOOK_PROBLEM
            matched_rules.append("problem_statement")

        contrast_patterns = {"but", "however", "surprisingly", "contrast_hi", "yet"}
        if any(p in seg.patterns for p in contrast_patterns):
            score += rcfg.HOOK_CONTRAST
            matched_rules.append("contrast")

        if seg.volume_delta > cfg.HOOK_ENERGY_VOLUME_DELTA and seg.speech_rate > cfg.HOOK_ENERGY_SPEECH_RATE:
            score += rcfg.HOOK_ENERGY_BONUS
            matched_rules.append("energy_escalation")

        if score > cfg.HOOK_MIN_SCORE:
            candidates.append(HookCandidate(
                segment_id=node_id,
                score=score,
                matched_rules=matched_rules
            ))

    return sorted(candidates, key=lambda c: c.score, reverse=True)
