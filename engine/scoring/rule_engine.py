import re
from ..data.models import CompleteClip, Segment
from ..config import get_config

_CONTEXT_PATTERNS = [
    r"(?i)as I (said|mentioned|told)",
    r"(?i)going back to",
    r"(?i)as we (discussed|talked)",
    r"(?i)earlier (I|we)",
    r"(?i)like I said",
    r"(?i)like I was (saying|saying earlier)",
    r"(?i)to go back",
    r"(?i)as mentioned (earlier|before)",
    r"(?i)coming back to",
    r"(?i)recall (that|when|how)",
    r"(?i)back to what I was",
    r"(?i)as I was",
]


def has_context_references(segments: list[Segment]) -> bool:
    for seg in segments:
        for pattern in _CONTEXT_PATTERNS:
            if re.search(pattern, seg.text):
                return True
    return False


def has_emotional_arc(segments: list[Segment]) -> bool:
    if len(segments) < 3:
        return False
    hook = segments[0]
    middle = segments[len(segments) // 2]
    ending = segments[-1]
    return (hook.sentiment < 0.1 and
            middle.sentiment < -0.1 and
            ending.sentiment > 0.2)


def validate_clip(clip: CompleteClip) -> tuple[bool, list[str], list[str]]:
    cfg = get_config().scoring
    passed = []
    failed = []

    if not (cfg.CLIP_MIN_DURATION <= clip.total_duration <= cfg.CLIP_MAX_DURATION):
        failed.append("total_duration_45_to_90")
    else:
        passed.append("total_duration_45_to_90")

    if clip.hook_duration > cfg.HOOK_MAX_DURATION:
        failed.append("hook_in_first_5_seconds")
    else:
        passed.append("hook_in_first_5_seconds")

    has_curiosity = any(
        any(p in s.patterns for p in ["curiosity", "what_if", "unknown", "biggest", "question", "unexpected"])
        for s in clip.segments
    )
    if has_curiosity:
        passed.append("has_curiosity")
    else:
        failed.append("has_curiosity")

    has_value = any(
        any(p in s.patterns for p in ["practicality", "steps", "lesson", "framework", "tip", "rule"])
        for s in clip.segments
    ) or any(abs(s.sentiment) > 0.3 for s in clip.segments)
    if has_value:
        passed.append("has_practicality_or_emotion")
    else:
        failed.append("has_practicality_or_emotion")

    if clip.speaker_changes <= 2:
        passed.append("max_2_speaker_changes")
    else:
        failed.append("max_2_speaker_changes")

    if not has_context_references(clip.segments):
        passed.append("no_context_gaps")
    else:
        failed.append("no_context_gaps")

    accepted = len(failed) == 0
    return accepted, passed, failed
