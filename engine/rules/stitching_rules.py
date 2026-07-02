from ..data.models import CompleteClip
from ..config import get_config


def compute_stitching_diversity(clip: CompleteClip) -> float:
    if len(clip.segments) < 3:
        return 0.0

    start_times = [s.start for s in clip.segments]
    max_gap = max(start_times) - min(start_times)

    cfg = get_config().scoring
    if max_gap > cfg.STITCHING_DIVERSITY_MAX:
        return 0.3
    elif max_gap > cfg.STITCHING_DIVERSITY_GOOD:
        return 1.0
    elif max_gap > cfg.STITCHING_DIVERSITY_MIN:
        return 0.7
    else:
        return 0.3
