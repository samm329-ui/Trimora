import re
import math
from collections import Counter
from ..data.models import CompleteClip, ValidClip, ScoredClip, Segment
from ..scoring.rule_engine import has_emotional_arc
from ..config import get_config


def make_valid_clip(clip: CompleteClip) -> ValidClip:
    return ValidClip(
        hook_segment=clip.segments[0],
        body_segments=clip.segments[1:-1],
        ending_segment=clip.segments[-1],
        all_segments=clip.segments,
        total_duration=clip.total_duration,
        hook_duration=clip.hook_duration,
        speaker_changes=clip.speaker_changes,
        diversity_score=clip.diversity_score,
    )


def score_clip(clip: ValidClip, full_transcript: str = "") -> ScoredClip:
    hook = clip.hook_segment
    hook_score = (
        (0.4 * (1.0 if any(p in hook.patterns for p in
                           ["curiosity", "what_if", "unknown", "biggest", "question", "unexpected"])
                else 0.0)) +
        (0.3 * min(hook.speech_rate / 3.0, 1.0)) +
        (0.3 * (1.0 - abs(hook.duration - 5.0) / 10.0))
    )

    body_score = 0.0
    for seg in clip.body_segments:
        body_score += (
            (0.3 * (1.0 if "personal" in seg.patterns else 0.0)) +
            (0.3 * (1.0 if abs(seg.sentiment) > 0.2 else 0.0)) +
            (0.4 * (1.0 if seg.duration < 12 else 0.5))
        )
    body_score /= max(len(clip.body_segments), 1)

    ending = clip.ending_segment
    ending_score = (
        (0.3 * max(ending.sentiment, 0.0)) +
        (0.4 * (1.0 if any(p in ending.patterns for p in
                           ["lesson", "practicality", "key_lesson", "action", "point", "framework"])
                else 0.0)) +
        (0.3 * (1.0 - abs(ending.duration - 7.0) / 10.0))
    )

    flow_score = compute_flow_score(clip)

    uniqueness_score = compute_uniqueness_score(clip, full_transcript) if full_transcript else 0.5

    total_score = (
        get_config().scoring.WEIGHT_HOOK * hook_score +
        get_config().scoring.WEIGHT_BODY * body_score +
        get_config().scoring.WEIGHT_ENDING * ending_score +
        get_config().scoring.WEIGHT_FLOW * flow_score +
        get_config().scoring.WEIGHT_PRACTICALITY * (
            1.0 if any("practicality" in s.patterns for s in clip.all_segments) else 0.0
        ) +
        get_config().scoring.WEIGHT_UNIQUENESS * uniqueness_score
    )

    return ScoredClip(
        clip=clip,
        hook_score=hook_score,
        body_score=body_score,
        ending_score=ending_score,
        flow_score=flow_score,
        total_score=total_score
    )


def compute_uniqueness_score(clip: ValidClip, full_transcript: str) -> float:
    all_words = re.findall(r'\w+', full_transcript.lower())
    if not all_words:
        return 0.5

    word_freq = Counter(all_words)
    n_total = len(all_words)

    clip_text = " ".join(s.text for s in clip.all_segments)
    clip_words = re.findall(r'\w+', clip_text.lower())

    if not clip_words:
        return 0.0

    idf_scores = []
    for w in clip_words:
        freq = word_freq.get(w, 0)
        idf = math.log((n_total + 1) / (freq + 1)) if freq > 0 else 0
        idf_scores.append(idf)

    avg_idf = sum(idf_scores) / len(idf_scores)
    return min(avg_idf / 3.0, 1.0)


def compute_flow_score(clip: ValidClip) -> float:
    score = 0.0

    hook_end = clip.hook_segment.end
    body_start = clip.body_segments[0].start
    gap1 = body_start - hook_end
    if gap1 < 2:
        score += 0.3
    elif gap1 < 10:
        score += 0.2

    body_end = clip.body_segments[-1].end
    ending_start = clip.ending_segment.start
    gap2 = ending_start - body_end
    if gap2 < 3:
        score += 0.3
    elif gap2 < 15:
        score += 0.2

    if has_emotional_arc(clip.all_segments):
        score += 0.4

    return min(score, 1.0)
