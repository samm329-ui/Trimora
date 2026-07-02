from ..data.models import Segment
from ..config import get_config


def compute_structural_features(segment: Segment, total_duration: float) -> dict:
    if total_duration <= 0:
        return {
            "position": 0.0,
            "recency": False,
            "is_first_30_pct": True,
            "is_last_30_pct": False,
        }

    cfg = get_config().scoring
    position = segment.start / total_duration

    return {
        "position": position,
        "recency": position > cfg.RECENCY_THRESHOLD,
        "is_first_30_pct": position < 0.3,
        "is_last_30_pct": position > cfg.RECENCY_THRESHOLD,
    }
