from ..data.models import HookCandidate, BodyCandidate, EndingCandidate, CompleteClip, Segment
from ..graph.knowledge_graph import KnowledgeGraph
from ..config import get_config


def _build_body_sequence(
    start_id: str,
    graph: KnowledgeGraph,
    target_duration: float,
    max_duration: float,
) -> list[Segment]:
    segs = []
    total = 0.0
    current_id = start_id
    visited = {start_id}

    sorted_nodes = sorted(
        [(n, graph.graph.nodes[n].get("start", 0)) for n in graph.graph.nodes()],
        key=lambda x: x[1]
    )
    node_positions = {nid: i for i, (nid, _) in enumerate(sorted_nodes)}

    while current_id and total < target_duration and total < max_duration:
        seg = graph.get_segment(current_id)
        if seg is None:
            break
        segs.append(seg)
        total += seg.duration

        if total >= target_duration or total >= max_duration:
            break

        pos = node_positions.get(current_id)
        if pos is None or pos + 1 >= len(sorted_nodes):
            break
        next_id = sorted_nodes[pos + 1][0]
        if next_id in visited:
            break
        visited.add(next_id)
        current_id = next_id

    return segs


def generate_candidates(
    hooks: list[HookCandidate],
    graph: KnowledgeGraph,
    max_hooks: int = 20,
    max_bodies_per_hook: int = 5,
    max_endings_per_body: int = 3
) -> list[CompleteClip]:
    cfg = get_config().scoring
    clips = []

    for hook in hooks[:max_hooks]:
        hook_seg = graph.get_segment(hook.segment_id)
        if hook_seg is None:
            continue

        from ..rules.body_rules import find_body_candidates
        bodies = find_body_candidates(hook.segment_id, graph)

        for body in bodies[:max_bodies_per_hook]:
            body_seg = graph.get_segment(body.id)
            if body_seg is None:
                continue
            if body_seg.start <= hook_seg.end:
                continue

            body_target = cfg.CLIP_MIN_DURATION - hook_seg.duration - 8.0
            body_max = cfg.CLIP_MAX_DURATION - hook_seg.duration - 3.0
            body_seq = _build_body_sequence(body.id, graph, max(15.0, body_target), body_max)

            if not body_seq:
                continue
            body_total = sum(s.duration for s in body_seq)
            body_ids = [s.id for s in body_seq]

            from ..rules.ending_rules import find_ending_candidates
            endings = find_ending_candidates(hook.segment_id, body_ids, graph)

            for ending in endings[:max_endings_per_body]:
                ending_seg = graph.get_segment(ending.id)
                if ending_seg is None:
                    continue
                if ending_seg.start <= body_seq[-1].start:
                    continue

                total = hook_seg.duration + body_total + ending_seg.duration
                if not (cfg.CLIP_MIN_DURATION <= total <= cfg.CLIP_MAX_DURATION):
                    continue

                clip = CompleteClip(
                    hook_id=hook.segment_id,
                    body_ids=body_ids,
                    ending_id=ending.id
                )
                clip.segments = [hook_seg, *body_seq, ending_seg]

                from ..rules.stitching_rules import compute_stitching_diversity
                clip.diversity_score = compute_stitching_diversity(clip)

                clips.append(clip)

    return clips
