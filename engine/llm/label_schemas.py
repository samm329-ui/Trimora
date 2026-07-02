from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class SegmentLabel:
    segment_id: str
    is_hook: float = 0.0
    hook_strength: float = 0.0
    is_context: float = 0.0
    is_takeaway: float = 0.0
    emotion: str = "neutral"
    requires_previous_context: float = 0.0
    creates_new_context: float = 0.0
    is_story: float = 0.0
    is_opinion: float = 0.0
    is_fact: float = 0.0
    speaker_confidence: float = 0.0
    saveability: float = 0.0
    shareability: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}


@dataclass
class CandidateLabel:
    candidate_id: str
    story_complete: float = 0.0
    transition_quality: float = 0.0
    context_missing: float = 0.0
    shareability: float = 0.0
    saveability: float = 0.0
    hook_strength: float = 0.0
    ending_strength: float = 0.0
    emotional_arc_build_up: float = 0.0
    naturalness: float = 0.0
    curiosity_gap: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}


@dataclass
class RejectionLabel:
    candidate_id: str
    reason: str = ""
    confidence: float = 0.0

    VALID_REASONS = [
        "missing_context",
        "weak_hook",
        "weak_ending",
        "no_emotional_arc",
        "transition_too_abrupt",
        "story_incomplete",
        "low_naturalness",
        "low_shareability",
    ]

    def is_valid_reason(self) -> bool:
        return self.reason in self.VALID_REASONS


@dataclass
class PatternLabel:
    pattern_id: str
    pattern_type: str = ""
    confidence: float = 0.0
    source_segment_ids: list[str] = field(default_factory=list)
    occurrence_count: int = 0
